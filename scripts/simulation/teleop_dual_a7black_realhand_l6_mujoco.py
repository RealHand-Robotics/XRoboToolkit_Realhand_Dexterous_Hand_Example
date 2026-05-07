"""Teleop dual A7black arms + dual RealHand L6 hands in MuJoCo."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import meshcat.transformations as tf
import mujoco
import numpy as np
import tyro
from dex_retargeting.constants import HandType, RetargetingType, RobotName

from sim_dual_a7black_realhand_l6_mujoco import (
    A7_ARM_JOINTS,
    A7_READY_JOINTS_RAD,
    write_combined_xml,
)
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import (
    MujocoTeleopController,
)
from xrobotoolkit_teleop.utils.dex_hand_utils import (
    DexHandTracker,
    pico_hand_state_to_mediapipe,
)
from xrobotoolkit_teleop.utils.geometry import apply_delta_pose
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

DEFAULT_A7BLACK_DUAL_URDF = os.path.join(ASSET_PATH, "real_hand/A7black/dual_A7black.urdf")
DEFAULT_LEFT_HAND_URDF_DIR = os.path.join(ASSET_PATH, "real_hand/l6/left")
DEFAULT_RIGHT_HAND_URDF_DIR = os.path.join(ASSET_PATH, "real_hand/l6/right")
DEFAULT_JOINT2_POLE_CLEARANCE_DEG = 10.0
DEFAULT_ARM_COMMAND_ALPHA = 0.35

HAND_CONTROL_JOINTS = [
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]
THUMB_ROLL_JOINT = "thunb_cmc_roll"  # spelling matches the URDF
THUMB_ROLL_DEFAULT = 0.6
JOINT_CLOSED_RAD = 1.25
PALM_JOINT_INDEX = 0
DEFAULT_HAND_TRACKING_ARM_SCALE = 0.3


def _build_initial_qpos(xml_path: str, ready_joints: dict[str, float]) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(xml_path)
    qpos = np.zeros(model.nq)
    for joint_name in A7_ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise RuntimeError(f"Joint not found in combined model: {joint_name}")
        qpos[model.jnt_qposadr[jid]] = ready_joints.get(joint_name, 0.0)
    return qpos


class DualA7blackRealhandL6TeleopController(MujocoTeleopController):
    HAND_PREFIXES = {"left_hand": "lh_", "right_hand": "rh_"}
    HAND_TRIGGERS = {"left_hand": "left_trigger", "right_hand": "right_trigger"}
    HAND_SIDES = {"left_hand": "left", "right_hand": "right"}

    def __init__(
        self,
        *args,
        left_hand_urdf_dir: str = DEFAULT_LEFT_HAND_URDF_DIR,
        right_hand_urdf_dir: str = DEFAULT_RIGHT_HAND_URDF_DIR,
        hand_tracking_arm_scale: float = DEFAULT_HAND_TRACKING_ARM_SCALE,
        arm_command_alpha: float = DEFAULT_ARM_COMMAND_ALPHA,
        **kwargs,
    ):
        self._hand_urdf_dirs = {
            "left_hand": left_hand_urdf_dir,
            "right_hand": right_hand_urdf_dir,
        }
        self.hand_tracking_arm_scale = float(hand_tracking_arm_scale)
        self.arm_command_alpha = float(np.clip(arm_command_alpha, 0.0, 1.0))
        super().__init__(*args, **kwargs)
        self._cache_a7_arm_actuator_ids()
        self._cache_hand_actuator_ids()
        self._init_hand_trackers()
        self._arm_input_mode = {name: None for name in self.manipulator_config}
        self._smoothed_arm_ctrl = self.mj_data.ctrl[self._a7_arm_actuator_ids].copy()

    def _cache_a7_arm_actuator_ids(self) -> None:
        ids = []
        for joint_name in A7_ARM_JOINTS:
            act_name = f"{joint_name}_pos"
            aid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if aid < 0:
                raise RuntimeError(f"A7black actuator not found: {act_name}")
            ids.append(aid)
        self._a7_arm_actuator_ids = np.asarray(ids, dtype=int)

    def _cache_hand_actuator_ids(self) -> None:
        self._hand_actuator_ids: dict[str, dict[str, int]] = {}
        self._thumb_roll_actuator_ids: dict[str, int] = {}
        for hand, prefix in self.HAND_PREFIXES.items():
            ids = {}
            for joint in HAND_CONTROL_JOINTS:
                act_name = f"{prefix}{joint}_pos"
                aid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                if aid < 0:
                    raise RuntimeError(f"Hand actuator not found: {act_name}")
                ids[joint] = aid
            self._hand_actuator_ids[hand] = ids

            roll_name = f"{prefix}{THUMB_ROLL_JOINT}_pos"
            roll_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, roll_name)
            if roll_id < 0:
                raise RuntimeError(f"Hand actuator not found: {roll_name}")
            self._thumb_roll_actuator_ids[hand] = roll_id

    def _init_hand_trackers(self) -> None:
        self._trackers: dict[str, DexHandTracker] = {}
        self._tracker_joint_names: dict[str, list[str]] = {}
        for hand, urdf_dir in self._hand_urdf_dirs.items():
            hand_type = HandType.left if self.HAND_SIDES[hand] == "left" else HandType.right
            tracker = DexHandTracker(
                robot_name=RobotName.real,
                urdf_path=urdf_dir,
                hand_type=hand_type,
                retargeting_type=RetargetingType.vector,
            )
            self._trackers[hand] = tracker
            self._tracker_joint_names[hand] = list(tracker.retargeting.joint_names)

    def _get_valid_hand_state(self, hand: str) -> np.ndarray | None:
        side = self.HAND_SIDES[hand]
        state = self.xr_client.get_hand_tracking_state(side)
        if state is None:
            return None
        state = np.asarray(state, dtype=float)
        if state.shape[0] <= PALM_JOINT_INDEX or state.shape[1] < 7:
            return None
        if np.allclose(state, 0.0) or not np.all(np.isfinite(state)):
            return None
        palm_pose = state[PALM_JOINT_INDEX]
        if np.allclose(palm_pose[:3], 0.0) or np.linalg.norm(palm_pose[3:7]) < 1e-6:
            return None
        return state

    def _retarget_targets(self, hand: str) -> dict[str, float] | None:
        state = self._get_valid_hand_state(hand)
        if state is None:
            return None
        mediapipe = pico_hand_state_to_mediapipe(state)
        qpos = self._trackers[hand].retarget(mediapipe)
        if qpos is None or len(qpos) != len(self._tracker_joint_names[hand]):
            return None
        return dict(zip(self._tracker_joint_names[hand], qpos))

    def _reset_arm_input_reference(self, src_name: str) -> None:
        self.ref_ee_xyz[src_name] = None
        self.ref_ee_quat[src_name] = None
        self.ref_controller_xyz[src_name] = None
        self.ref_controller_quat[src_name] = None

    def _select_arm_input_pose(self, src_name: str, config: dict) -> tuple[bool, np.ndarray | None]:
        xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
        if xr_grip_val > 0.9:
            return False, self.xr_client.get_pose_by_name(config["pose_source"])
        hand_state = self._get_valid_hand_state(src_name)
        if hand_state is not None:
            return True, hand_state[PALM_JOINT_INDEX]
        return False, None

    def _update_ik(self) -> None:
        self._update_robot_state()
        self.placo_robot.update_kinematics()

        for src_name, config in self.manipulator_config.items():
            using_hand_tracking, input_pose = self._select_arm_input_pose(src_name, config)
            self.active[src_name] = input_pose is not None
            input_mode = "hand_tracking" if using_hand_tracking else "controller"

            if self.active[src_name]:
                if self._arm_input_mode[src_name] != input_mode:
                    self._reset_arm_input_reference(src_name)
                    self._arm_input_mode[src_name] = input_mode

                if self.ref_ee_xyz[src_name] is None:
                    print(f"{src_name} is activated with {input_mode}.")
                    self.ref_ee_xyz[src_name], self.ref_ee_quat[src_name] = self._get_link_pose(config["link_name"])

                delta_xyz, delta_rot = self._process_xr_pose(input_pose, src_name)
                if using_hand_tracking:
                    delta_xyz = delta_xyz * self.hand_tracking_arm_scale

                if using_hand_tracking:
                    target_xyz = self.ref_ee_xyz[src_name] + delta_xyz
                    target_pose = tf.quaternion_matrix(self.ref_ee_quat[src_name])
                    target_pose[:3, 3] = target_xyz
                    self.effector_task[src_name].T_world_frame = target_pose
                else:
                    target_xyz, target_quat = apply_delta_pose(
                        self.ref_ee_xyz[src_name],
                        self.ref_ee_quat[src_name],
                        delta_xyz,
                        delta_rot,
                    )
                    target_pose = tf.quaternion_matrix(target_quat)
                    target_pose[:3, 3] = target_xyz
                    self.effector_task[src_name].T_world_frame = target_pose
            else:
                if self.ref_ee_xyz[src_name] is not None:
                    print(f"{src_name} is deactivated.")
                self._reset_arm_input_reference(src_name)
                self._arm_input_mode[src_name] = None

        try:
            self.solver.solve(True)
        except RuntimeError as e:
            print(f"IK solver failed: {e}")

    def _apply_trigger_fallback(self, hand: str) -> None:
        trigger = self.xr_client.get_key_value_by_name(self.HAND_TRIGGERS[hand])
        close_target = float(np.clip(trigger, 0.0, 1.0)) * JOINT_CLOSED_RAD
        for aid in self._hand_actuator_ids[hand].values():
            self.mj_data.ctrl[aid] = close_target
        self.mj_data.ctrl[self._thumb_roll_actuator_ids[hand]] = THUMB_ROLL_DEFAULT

    def _apply_retarget(self, hand: str, targets: dict[str, float]) -> None:
        for joint, aid in self._hand_actuator_ids[hand].items():
            if joint in targets:
                self.mj_data.ctrl[aid] = float(targets[joint])
        roll_aid = self._thumb_roll_actuator_ids[hand]
        self.mj_data.ctrl[roll_aid] = float(targets.get(THUMB_ROLL_JOINT, THUMB_ROLL_DEFAULT))

    def _update_hands(self) -> None:
        for hand in self.HAND_PREFIXES:
            targets = self._retarget_targets(hand)
            if targets is not None:
                self._apply_retarget(hand, targets)
            else:
                self._apply_trigger_fallback(hand)

    def _send_command(self) -> None:
        super()._send_command()
        self._smooth_arm_command()
        self._update_hands()

    def _smooth_arm_command(self) -> None:
        if self.arm_command_alpha >= 1.0:
            return

        desired = self.mj_data.ctrl[self._a7_arm_actuator_ids].copy()
        alpha = self.arm_command_alpha
        self._smoothed_arm_ctrl = (1.0 - alpha) * self._smoothed_arm_ctrl + alpha * desired
        self.mj_data.ctrl[self._a7_arm_actuator_ids] = self._smoothed_arm_ctrl


def _ensure_combined_xml(xml_path: str) -> str:
    out = Path(xml_path) if xml_path else Path(tempfile.gettempdir()) / "dual_a7black_realhand_l6.xml"
    write_combined_xml(out)
    return str(out)


def _set_a7_joint2_pole_limits(controller, clearance_deg: float) -> None:
    if clearance_deg <= 0.0:
        return

    clearance = np.deg2rad(float(clearance_deg))
    controller.placo_robot.set_joint_limits("left_joint2", clearance, 3.2)
    controller.placo_robot.set_joint_limits("right_joint2", -3.2, -clearance)


def _a7_ready_joints(clearance_deg: float) -> dict[str, float]:
    ready_joints = dict(A7_READY_JOINTS_RAD)
    if clearance_deg > 0.0:
        clearance = np.deg2rad(float(clearance_deg))
        ready_joints["left_joint2"] = max(ready_joints.get("left_joint2", 0.0), clearance)
        ready_joints["right_joint2"] = min(ready_joints.get("right_joint2", 0.0), -clearance)
    return ready_joints


def main(
    xml_path: str = "",
    robot_urdf_path: str = DEFAULT_A7BLACK_DUAL_URDF,
    left_hand_urdf_dir: str = DEFAULT_LEFT_HAND_URDF_DIR,
    right_hand_urdf_dir: str = DEFAULT_RIGHT_HAND_URDF_DIR,
    scale_factor: float = 1.0,
    hand_tracking_arm_scale: float = DEFAULT_HAND_TRACKING_ARM_SCALE,
    arm_command_alpha: float = DEFAULT_ARM_COMMAND_ALPHA,
    visualize_placo: bool = False,
    enable_velocity_limits: bool = False,
    joint2_pole_clearance_deg: float = DEFAULT_JOINT2_POLE_CLEARANCE_DEG,
):
    """Run dual A7black + dual RealHand L6 teleoperation in MuJoCo."""
    combined_xml = _ensure_combined_xml(xml_path)
    ready_joints = _a7_ready_joints(joint2_pole_clearance_deg)
    config = {
        "left_hand": {
            "link_name": "left_link8",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "vis_target": "left_target",
        },
        "right_hand": {
            "link_name": "right_link8",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "vis_target": "right_target",
        },
    }

    controller = DualA7blackRealhandL6TeleopController(
        xml_path=combined_xml,
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        visualize_placo=visualize_placo,
        mj_qpos_init=_build_initial_qpos(combined_xml, ready_joints),
        left_hand_urdf_dir=left_hand_urdf_dir,
        right_hand_urdf_dir=right_hand_urdf_dir,
        hand_tracking_arm_scale=hand_tracking_arm_scale,
        arm_command_alpha=arm_command_alpha,
    )
    _set_a7_joint2_pole_limits(controller, joint2_pole_clearance_deg)

    joints_task = controller.solver.add_joints_task()
    joints_task.set_joints(
        {
            joint: ready_joints.get(joint, 0.0)
            for joint in controller.placo_robot.joint_names()
        }
    )
    joints_task.configure("joints_regularization", "soft", 1e-4)
    if enable_velocity_limits:
        controller.solver.enable_velocity_limits(True)

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)

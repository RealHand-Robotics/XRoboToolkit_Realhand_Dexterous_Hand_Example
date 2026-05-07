"""Teleop for dual UR5e arms + dual RealHand L6 hands in MuJoCo.

Controls:
    - left/right GRIP button  -> when hand tracking is unavailable, activate the
                                 corresponding UR5e arm and follow the
                                 controller pose (existing dual UR5e behavior).
    - left/right HAND TRACKING (when active and producing valid joints)
                              -> drives both the UR5e end effector from the
                                 palm position and the RealHand L6 fingers via
                                 ``dex_retargeting``. The UR5e keeps its
                                 current end-effector orientation in this mode.
    - left/right TRIGGER      -> fallback hand open/close command when hand
                                 tracking is unavailable for that side.
                                 0.0 = open, 1.0 = power-grip.

Hand tracking takes precedence per side: as soon as the XR client reports an
active, non-zero hand pose, the UR5e target is driven from the palm root and
the trigger value is ignored for that hand.

The combined arm + hand model is built at startup by
``sim_dual_ur5e_realhand_l6_mujoco.build_combined_spec()`` (UR5e MJCF + L6
URDFs attached at each ``<side>_attachment_site``) and written to a temp
file. The dual_ur5e.urdf is reused for IK -- placo only sees the 12 arm
joints, so the hand DOFs are driven directly from this controller.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mujoco
import meshcat.transformations as tf
import numpy as np
import tyro
from dex_retargeting.constants import HandType, RetargetingType, RobotName

from sim_dual_ur5e_realhand_l6_mujoco import write_combined_xml
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import (
    MujocoTeleopController,
)
from xrobotoolkit_teleop.utils.dex_hand_utils import (
    DexHandTracker,
    pico_hand_state_to_mediapipe,
)
from xrobotoolkit_teleop.utils.geometry import apply_delta_pose
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

DEFAULT_LEFT_HAND_URDF_DIR = os.path.join(ASSET_PATH, "real_hand/l6/left")
DEFAULT_RIGHT_HAND_URDF_DIR = os.path.join(ASSET_PATH, "real_hand/l6/right")

HAND_CONTROL_JOINTS = [
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]
THUMB_ROLL_JOINT = "thunb_cmc_roll"  # spelling matches the URDF
THUMB_ROLL_DEFAULT = 0.6  # rad of opposition while teleop is running
PALM_JOINT_INDEX = 0  # PICO/OpenXR palm root pose, used as the arm teleop pose in hand-tracking mode
DEFAULT_HAND_TRACKING_ARM_SCALE = 0.3  # extra gain applied on top of scale_factor for calmer palm-driven arm motion

JOINT_CLOSED_RAD = 1.25  # close target for the four MCPs and thumb pitch (just under their 1.26 limit)

LEFT_INITIAL_JOINT_DEG = np.array([165.26, -47.50, 118.93, -38.96, 87.51, 149.56])
RIGHT_INITIAL_JOINT_DEG = np.array([-166.53, -136.17, -106.02, 210.01, -86.87, -128.40])

ARM_INITIAL_JOINTS_RAD = {
    "left_shoulder_pan_joint": np.deg2rad(LEFT_INITIAL_JOINT_DEG[0]),
    "left_shoulder_lift_joint": np.deg2rad(LEFT_INITIAL_JOINT_DEG[1]),
    "left_elbow_joint": np.deg2rad(LEFT_INITIAL_JOINT_DEG[2]),
    "left_wrist_1_joint": np.deg2rad(LEFT_INITIAL_JOINT_DEG[3]),
    "left_wrist_2_joint": np.deg2rad(LEFT_INITIAL_JOINT_DEG[4]),
    "left_wrist_3_joint": np.deg2rad(LEFT_INITIAL_JOINT_DEG[5]),
    "right_shoulder_pan_joint": np.deg2rad(RIGHT_INITIAL_JOINT_DEG[0]),
    "right_shoulder_lift_joint": np.deg2rad(RIGHT_INITIAL_JOINT_DEG[1]),
    "right_elbow_joint": np.deg2rad(RIGHT_INITIAL_JOINT_DEG[2]),
    "right_wrist_1_joint": np.deg2rad(RIGHT_INITIAL_JOINT_DEG[3]),
    "right_wrist_2_joint": np.deg2rad(RIGHT_INITIAL_JOINT_DEG[4]),
    "right_wrist_3_joint": np.deg2rad(RIGHT_INITIAL_JOINT_DEG[5]),
}


def _build_initial_qpos(xml_path: str) -> np.ndarray:
    """Compose a full nq qpos vector with the requested arm pose and zeros elsewhere."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    qpos = np.zeros(model.nq)
    for joint_name, value in ARM_INITIAL_JOINTS_RAD.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise RuntimeError(f"Joint not found in combined model: {joint_name}")
        qpos[model.jnt_qposadr[jid]] = value
    return qpos


class DualUR5eRealhandL6TeleopController(MujocoTeleopController):
    """Hand-tracking-first hand control on top of the standard dual-arm IK.

    For each side, the per-frame hand command is chosen as:
      1. If XR hand tracking is active and returns a non-zero pose, retarget
         the human pose with ``dex_retargeting`` and write the resulting joint
         angles into the L6 actuators.
      2. Otherwise, fall back to a linear open<->closed interpolation driven
         by that side's trigger value.
    """

    HAND_PREFIXES = {"left_hand": "lh_", "right_hand": "rh_"}
    HAND_TRIGGERS = {"left_hand": "left_trigger", "right_hand": "right_trigger"}
    HAND_SIDES = {"left_hand": "left", "right_hand": "right"}

    def __init__(
        self,
        *args,
        left_hand_urdf_dir: str = DEFAULT_LEFT_HAND_URDF_DIR,
        right_hand_urdf_dir: str = DEFAULT_RIGHT_HAND_URDF_DIR,
        hand_tracking_arm_scale: float = DEFAULT_HAND_TRACKING_ARM_SCALE,
        **kwargs,
    ):
        self._hand_urdf_dirs = {
            "left_hand": left_hand_urdf_dir,
            "right_hand": right_hand_urdf_dir,
        }
        self.hand_tracking_arm_scale = float(hand_tracking_arm_scale)
        super().__init__(*args, **kwargs)
        self._cache_hand_actuator_ids()
        self._init_hand_trackers()
        self._arm_input_mode = {name: None for name in self.manipulator_config}

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

    def _retarget_targets(self, hand: str) -> dict[str, float] | None:
        state = self._get_valid_hand_state(hand)
        if state is None:
            return None
        mediapipe = pico_hand_state_to_mediapipe(state)
        qpos = self._trackers[hand].retarget(mediapipe)
        if qpos is None or len(qpos) != len(self._tracker_joint_names[hand]):
            return None
        return dict(zip(self._tracker_joint_names[hand], qpos))

    def _get_valid_hand_state(self, hand: str) -> np.ndarray | None:
        side = self.HAND_SIDES[hand]
        state = self.xr_client.get_hand_tracking_state(side)
        if state is None:
            return None
        state = np.asarray(state, dtype=float)
        if state.shape[0] <= PALM_JOINT_INDEX or state.shape[1] < 7:
            return None
        if np.allclose(state, 0.0):
            return None
        if not np.all(np.isfinite(state)):
            return None
        palm_pose = state[PALM_JOINT_INDEX]
        if np.allclose(palm_pose[:3], 0.0) or np.linalg.norm(palm_pose[3:7]) < 1e-6:
            return None
        return state

    def _reset_arm_input_reference(self, src_name: str) -> None:
        self.ref_ee_xyz[src_name] = None
        self.ref_ee_quat[src_name] = None
        self.ref_controller_xyz[src_name] = None
        self.ref_controller_quat[src_name] = None

    def _select_arm_input_pose(self, src_name: str, config: dict) -> tuple[bool, np.ndarray | None]:
        hand_state = self._get_valid_hand_state(src_name)
        if hand_state is not None:
            return True, hand_state[PALM_JOINT_INDEX]
        xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
        if xr_grip_val > 0.9:
            return False, self.xr_client.get_pose_by_name(config["pose_source"])
        return False, None

    def _update_ik(self):
        """Use palm-root teleop in hand-tracking mode, otherwise fall back to controller grip teleop."""
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

                if self.effector_control_mode[src_name] == "position":
                    target_xyz = self.ref_ee_xyz[src_name] + delta_xyz
                    self.effector_task[src_name].target_world = target_xyz
                elif using_hand_tracking:
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

        self._update_motion_tracker_tasks()

        try:
            self.solver.solve(True)
        except RuntimeError as e:
            print(f"IK solver failed: {e}")

    def _apply_trigger_fallback(self, hand: str) -> None:
        t = float(np.clip(self.xr_client.get_key_value_by_name(self.HAND_TRIGGERS[hand]), 0.0, 1.0))
        close_target = t * JOINT_CLOSED_RAD
        for aid in self._hand_actuator_ids[hand].values():
            self.mj_data.ctrl[aid] = close_target
        self.mj_data.ctrl[self._thumb_roll_actuator_ids[hand]] = THUMB_ROLL_DEFAULT

    def _apply_retarget(self, hand: str, targets: dict[str, float]) -> None:
        for joint, aid in self._hand_actuator_ids[hand].items():
            if joint in targets:
                self.mj_data.ctrl[aid] = float(targets[joint])
        roll_aid = self._thumb_roll_actuator_ids[hand]
        if THUMB_ROLL_JOINT in targets:
            self.mj_data.ctrl[roll_aid] = float(targets[THUMB_ROLL_JOINT])
        else:
            self.mj_data.ctrl[roll_aid] = THUMB_ROLL_DEFAULT

    def _update_hands(self) -> None:
        for hand in self.HAND_PREFIXES:
            targets = self._retarget_targets(hand)
            if targets is not None:
                self._apply_retarget(hand, targets)
            else:
                self._apply_trigger_fallback(hand)

    def _send_command(self):
        super()._send_command()
        self._update_hands()


def _ensure_combined_xml(xml_path: str) -> str:
    out = Path(xml_path) if xml_path else Path(tempfile.gettempdir()) / "dual_ur5e_realhand_l6.xml"
    write_combined_xml(out)
    return str(out)


def main(
    xml_path: str = "",
    robot_urdf_path: str = os.path.join(ASSET_PATH, "universal_robots_ur5e/dual_ur5e.urdf"),
    left_hand_urdf_dir: str = DEFAULT_LEFT_HAND_URDF_DIR,
    right_hand_urdf_dir: str = DEFAULT_RIGHT_HAND_URDF_DIR,
    scale_factor: float = 1.5,
    hand_tracking_arm_scale: float = DEFAULT_HAND_TRACKING_ARM_SCALE,
    visualize_placo: bool = False,
):
    """Run the dual UR5e + dual RealHand L6 teleoperation in MuJoCo.

    Args:
        xml_path: Path to write the combined arm+hand MJCF. If empty, a path
            in the system temp directory is used.
        robot_urdf_path: URDF used for the placo IK solver (arm joints only).
        left_hand_urdf_dir: Directory containing the left RealHand L6 URDF
            used by ``dex_retargeting``.
        right_hand_urdf_dir: Directory containing the right RealHand L6 URDF
            used by ``dex_retargeting``.
        scale_factor: Controller-to-arm motion scaling.
        hand_tracking_arm_scale: Extra scale for palm-driven arm translation in
            hand-tracking mode. Lower values reduce sensitivity.
        visualize_placo: Open the placo meshcat viewer.
    """
    combined_xml = _ensure_combined_xml(xml_path)

    config = {
        "right_hand": {
            "link_name": "right_tool0",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "vis_target": "right_target",
        },
        "left_hand": {
            "link_name": "left_tool0",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "vis_target": "left_target",
        },
    }

    controller = DualUR5eRealhandL6TeleopController(
        xml_path=combined_xml,
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        visualize_placo=visualize_placo,
        mj_qpos_init=_build_initial_qpos(combined_xml),
        left_hand_urdf_dir=left_hand_urdf_dir,
        right_hand_urdf_dir=right_hand_urdf_dir,
        hand_tracking_arm_scale=hand_tracking_arm_scale,
    )

    joints_task = controller.solver.add_joints_task()
    joints_task.set_joints({joint: 0.0 for joint in controller.placo_robot.joint_names()})
    joints_task.configure("joints_regularization", "soft", 1e-4)

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)

import os
import time
from typing import Dict, Literal, Optional

import numpy as np

from xrobotoolkit_teleop.common.base_hardware_teleop_controller import (
    HardwareTeleopController,
)
from xrobotoolkit_teleop.hardware.interface.a7black import A7BlackInterface
from xrobotoolkit_teleop.hardware.interface.realsense import RealSenseCameraInterface
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


DEFAULT_A7BLACK_URDF_PATH = os.path.join(ASSET_PATH, "real_hand/A7black/A7black.urdf")
DEFAULT_DUAL_A7BLACK_URDF_PATH = os.path.join(ASSET_PATH, "real_hand/A7black/dual_A7black.urdf")
DEFAULT_SCALE_FACTOR = 1.0

DEFAULT_A7BLACK_CAN_PORTS = {
    "left_arm": "can0",
    "right_arm": "can1",
}
DEFAULT_A7BLACK_ARM_SIDES = {
    "left_arm": "left",
    "right_arm": "right",
}
DEFAULT_JOINT2_POLE_CLEARANCE_DEG = 10.0

DEFAULT_A7BLACK_MANIPULATOR_CONFIG = {
    "right_arm": {
        "link_name": "right_link8",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
    },
}

DEFAULT_DUAL_A7BLACK_MANIPULATOR_CONFIG = {
    "left_arm": {
        "link_name": "left_link8",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
    },
    "right_arm": {
        "link_name": "right_link8",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
    },
}

DEFAULT_RIGHT_WRIST_CAM_SERIAL = "218622272014"
DEFAULT_LEFT_WRIST_CAM_SERIAL = "218622272499"
DEFAULT_BASE_CAM_SERIAL = "215222077461"

CAM_SERIAL_DICT = {
    "left_wrist": DEFAULT_LEFT_WRIST_CAM_SERIAL,
    "right_wrist": DEFAULT_RIGHT_WRIST_CAM_SERIAL,
    "base": DEFAULT_BASE_CAM_SERIAL,
}


class DualA7BlackTeleopController(HardwareTeleopController):
    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_DUAL_A7BLACK_URDF_PATH,
        manipulator_config: dict = DEFAULT_DUAL_A7BLACK_MANIPULATOR_CONFIG,
        can_ports: Dict[str, str] = DEFAULT_A7BLACK_CAN_PORTS,
        arm_sides: Dict[str, Literal["left", "right"]] = DEFAULT_A7BLACK_ARM_SIDES,
        interface_type: str = "socketcan",
        world_frame: Literal["urdf", "maestro"] = "urdf",
        tcp_offsets: Optional[Dict[str, list[float]]] = None,
        arm_velocity: float = 1.0,
        arm_acceleration: float = 10.0,
        enable_on_start: bool = True,
        home_on_start: bool = False,
        home_timeout_sec: float = 20.0,
        home_tolerance_deg: float = 2.0,
        check_joint_limits: bool = True,
        use_streaming_setpoint: bool = True,
        joint2_pole_clearance_deg: float = DEFAULT_JOINT2_POLE_CLEARANCE_DEG,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        visualize_placo: bool = False,
        control_rate_hz: int = 50,
        enable_log_data: bool = True,
        log_dir: str = "logs/dual_a7black",
        log_freq: float = 50,
        enable_camera: bool = False,
        camera_serial_dict: Dict[str, str] = CAM_SERIAL_DICT,
        camera_width: int = 424,
        camera_height: int = 240,
        camera_fps: int = 60,
        enable_camera_depth: bool = False,
        enable_camera_compression: bool = True,
        camera_jpg_quality: int = 85,
    ):
        self.can_ports = dict(can_ports)
        self.arm_sides = dict(arm_sides)
        self.interface_type = interface_type
        self.world_frame = world_frame
        self.tcp_offsets = dict(tcp_offsets) if tcp_offsets is not None else {}
        self.arm_velocity = float(arm_velocity)
        self.arm_acceleration = float(arm_acceleration)
        self.enable_on_start = enable_on_start
        self.home_on_start = home_on_start
        self.home_timeout_sec = float(home_timeout_sec)
        self.home_tolerance_deg = float(home_tolerance_deg)
        self.check_joint_limits = check_joint_limits
        self.use_streaming_setpoint = use_streaming_setpoint
        self.joint2_pole_clearance_deg = float(joint2_pole_clearance_deg)

        self.camera_serial_dict = camera_serial_dict
        self.camera_serial_to_name = {serial: name for name, serial in camera_serial_dict.items()}
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.enable_camera_depth = enable_camera_depth
        self.enable_camera_compression = enable_camera_compression
        self.camera_jpg_quality = camera_jpg_quality

        self._robot_setup_done = False
        self._last_command_error_time: Dict[str, float] = {}

        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            R_headset_world=R_headset_world,
            floating_base=False,
            scale_factor=scale_factor,
            visualize_placo=visualize_placo,
            control_rate_hz=control_rate_hz,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
            enable_camera=enable_camera,
            camera_fps=camera_fps,
        )

    def _placo_setup(self):
        super()._placo_setup()
        self.placo_arm_joint_slice: Dict[str, slice] = {}
        self.placo_arm_joint_names: Dict[str, list[str]] = {}
        for arm_name, config in self.manipulator_config.items():
            ee_link_name = config["link_name"]
            arm_prefix = ee_link_name.replace("link8", "")
            arm_joint_names = [f"{arm_prefix}joint{i}" for i in range(1, 8)]
            self.placo_arm_joint_names[arm_name] = arm_joint_names
            self.placo_arm_joint_slice[arm_name] = slice(
                self.placo_robot.get_joint_offset(arm_joint_names[0]),
                self.placo_robot.get_joint_offset(arm_joint_names[-1]) + 1,
            )
        self._set_placo_joint2_pole_limits()

    def _set_placo_joint2_pole_limits(self) -> None:
        if self.joint2_pole_clearance_deg <= 0.0:
            return

        clearance = np.deg2rad(self.joint2_pole_clearance_deg)
        for arm_name, side in self.arm_sides.items():
            if arm_name not in self.manipulator_config:
                continue
            joint_name = f"{side}_joint2"
            if side == "left":
                self.placo_robot.set_joint_limits(joint_name, clearance, 3.2)
            else:
                self.placo_robot.set_joint_limits(joint_name, -3.2, -clearance)

    def _robot_setup(self):
        if self._robot_setup_done:
            return

        self.arm_controllers: Dict[str, A7BlackInterface] = {}
        for arm_name, can_port in self.can_ports.items():
            if arm_name not in self.manipulator_config:
                continue

            side = self.arm_sides.get(arm_name)
            if side is None:
                raise ValueError(f"Missing A7 side for {arm_name}")

            print(f"Setting up A7black {arm_name} ({side}) on CAN port: {can_port}")
            arm = A7BlackInterface(
                can_port=can_port,
                side=side,
                interface_type=self.interface_type,
                world_frame=self.world_frame,
                tcp_offset=self.tcp_offsets.get(arm_name),
                default_velocity=self.arm_velocity,
                default_acceleration=self.arm_acceleration,
                enable_on_start=self.enable_on_start,
                check_joint_limits=self.check_joint_limits,
                use_streaming_setpoint=self.use_streaming_setpoint,
                joint2_pole_clearance_deg=self.joint2_pole_clearance_deg,
                home_timeout_sec=self.home_timeout_sec,
                home_tolerance_deg=self.home_tolerance_deg,
            )
            self.arm_controllers[arm_name] = arm

        self._robot_setup_done = True

        if self.home_on_start:
            print("Moving A7black arms home...")
            for arm in self.arm_controllers.values():
                arm.go_home(blocking=True)
            print("A7black arms are home.")

    def _initialize_camera(self):
        if self.enable_camera:
            print("Initializing camera...")
            try:
                self.camera_interface = RealSenseCameraInterface(
                    width=self.camera_width,
                    height=self.camera_height,
                    fps=self.camera_fps,
                    serial_numbers=list(self.camera_serial_dict.values()),
                    enable_depth=self.enable_camera_depth,
                    enable_compression=self.enable_camera_compression,
                    jpg_quality=self.camera_jpg_quality,
                )
                self.camera_interface.start()
                print("Camera initialized successfully.")
            except Exception as e:
                print(f"Error initializing camera: {e}")
                self.camera_interface = None

    def _update_robot_state(self):
        for arm_name, controller in self.arm_controllers.items():
            q_slice = self.placo_arm_joint_slice[arm_name]
            self.placo_robot.state.q[q_slice] = controller.get_urdf_joint_positions()

    def _send_command(self):
        for arm_name, controller in self.arm_controllers.items():
            if not self.active.get(arm_name, False):
                continue

            q_des = self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]].copy()
            try:
                controller.set_urdf_joint_positions(q_des, blocking=False)
            except Exception as exc:
                self._report_command_error(arm_name, exc)

    def _report_command_error(self, arm_name: str, exc: Exception) -> None:
        now = time.time()
        last = self._last_command_error_time.get(arm_name, 0.0)
        if now - last > 1.0:
            print(f"[A7black {arm_name}] command error: {exc}")
            self._last_command_error_time[arm_name] = now

    def _get_robot_state_for_logging(self) -> Dict:
        return {
            "qpos": {
                arm: c.get_urdf_joint_positions() for arm, c in self.arm_controllers.items()
            },
            "qpos_sdk": {arm: c.get_joint_positions() for arm, c in self.arm_controllers.items()},
            "qvel": {
                arm: c.get_urdf_joint_velocities() for arm, c in self.arm_controllers.items()
            },
            "qvel_sdk": {arm: c.get_joint_velocities() for arm, c in self.arm_controllers.items()},
            "qpos_des": {
                arm: self.placo_robot.state.q[self.placo_arm_joint_slice[arm]].copy()
                for arm in self.arm_controllers
            },
        }

    def _get_camera_frame_for_logging(self) -> Dict:
        if not self.camera_interface:
            return {}

        if self.camera_interface.enable_compression:
            frames_by_serial = self.camera_interface.get_compressed_frames()
        else:
            frames_by_serial = self.camera_interface.get_frames()

        if not frames_by_serial:
            return {}

        frames_by_name = {}
        for serial, frames in frames_by_serial.items():
            camera_name = self.camera_serial_to_name.get(serial, serial)
            frames_by_name[camera_name] = frames

        return frames_by_name

    def reset(self, blocking: bool = True) -> bool:
        self._robot_setup()
        print("Moving A7black arms home...")
        success = True
        for controller in self.arm_controllers.values():
            success = controller.go_home(blocking=blocking) and success
        if blocking and success:
            print("A7black arms are home.")
        elif blocking:
            print("A7black home command timed out before all arms reached target.")
        return success

    def close(self, disable: bool = True) -> None:
        self._shutdown_robot(disable=disable)

    def _shutdown_robot(self, disable: bool = True):
        for arm_name, controller in getattr(self, "arm_controllers", {}).items():
            try:
                action = "Disabling" if disable else "Closing connection to"
                print(f"{action} A7black {arm_name}")
                controller.close(disable=disable)
            except Exception as exc:
                print(f"[A7black {arm_name}] shutdown error: {exc}")

import threading
from typing import Optional

import tyro

from xrobotoolkit_teleop.hardware.a7black_teleop_controller import (
    DEFAULT_DUAL_A7BLACK_MANIPULATOR_CONFIG,
    DEFAULT_DUAL_A7BLACK_URDF_PATH,
    DEFAULT_JOINT2_POLE_CLEARANCE_DEG,
    DualA7BlackTeleopController,
)
from xrobotoolkit_teleop.hardware.realhand_l6_controller import (
    DEFAULT_LEFT_URDF,
    DEFAULT_RIGHT_URDF,
    DualRealHandL6Controller,
)
from xrobotoolkit_teleop.hardware.realhand_hardware_map import (
    DEFAULT_REALHAND_HARDWARE_MAP_PATH,
    resolve_realhand_can_ports,
    validate_realhand_can_ports,
)


def main(
    robot_urdf_path: str = DEFAULT_DUAL_A7BLACK_URDF_PATH,
    left_arm_can_port: Optional[str] = None,
    right_arm_can_port: Optional[str] = None,
    left_hand_can_port: Optional[str] = None,
    right_hand_can_port: Optional[str] = None,
    hardware_map_path: str = str(DEFAULT_REALHAND_HARDWARE_MAP_PATH),
    left_hand_urdf: str = DEFAULT_LEFT_URDF,
    right_hand_urdf: str = DEFAULT_RIGHT_URDF,
    interface_type: str = "socketcan",
    world_frame: str = "urdf",
    scale_factor: float = 1.0,
    arm_velocity: float = 1.0,
    arm_acceleration: float = 10.0,
    joint2_pole_clearance_deg: float = DEFAULT_JOINT2_POLE_CLEARANCE_DEG,
    home_timeout_sec: float = 20.0,
    home_tolerance_deg: float = 2.0,
    hand_control_hz: float = 60.0,
    enable_on_start: bool = True,
    home_on_start: bool = False,
    reset: bool = False,
    disable_after_reset: bool = False,
    enable_camera: bool = False,
    enable_log_data: bool = True,
    visualize_placo: bool = True,
    control_rate_hz: int = 50,
    log_dir: str = "logs/dual_a7black_realhand_l6",
) -> None:
    """Run dual A7black arms and dual RealHand L6 hands on hardware."""
    arm_controller = None
    hand_controller = None
    can_ports = resolve_realhand_can_ports(
        hardware_map_path,
        required_roles=["left_arm", "right_arm", "left_hand", "right_hand"],
    )
    left_arm_can_port = left_arm_can_port or can_ports["left_arm"]
    right_arm_can_port = right_arm_can_port or can_ports["right_arm"]
    left_hand_can_port = left_hand_can_port or can_ports["left_hand"]
    right_hand_can_port = right_hand_can_port or can_ports["right_hand"]
    errors = validate_realhand_can_ports(
        {
            "left_arm": left_arm_can_port,
            "right_arm": right_arm_can_port,
            "left_hand": left_hand_can_port,
            "right_hand": right_hand_can_port,
        },
        ["left_arm", "right_arm", "left_hand", "right_hand"],
    )
    if errors:
        raise ValueError("Invalid CAN mapping:\n  - " + "\n  - ".join(errors))
    print(
        "CAN mapping: "
        f"left_arm={left_arm_can_port}, right_arm={right_arm_can_port}, "
        f"left_hand={left_hand_can_port}, right_hand={right_hand_can_port}"
    )

    try:
        arm_controller = DualA7BlackTeleopController(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=DEFAULT_DUAL_A7BLACK_MANIPULATOR_CONFIG,
            can_ports={
                "left_arm": left_arm_can_port,
                "right_arm": right_arm_can_port,
            },
            arm_sides={
                "left_arm": "left",
                "right_arm": "right",
            },
            interface_type=interface_type,
            world_frame=world_frame,
            scale_factor=scale_factor,
            arm_velocity=arm_velocity,
            arm_acceleration=arm_acceleration,
            joint2_pole_clearance_deg=joint2_pole_clearance_deg,
            home_timeout_sec=home_timeout_sec,
            home_tolerance_deg=home_tolerance_deg,
            enable_on_start=enable_on_start,
            home_on_start=home_on_start,
            enable_camera=enable_camera,
            enable_log_data=enable_log_data,
            visualize_placo=visualize_placo,
            control_rate_hz=control_rate_hz,
            log_dir=log_dir,
        )
        hand_controller = DualRealHandL6Controller(
            xr_client=arm_controller.xr_client,
            left_interface=left_hand_can_port,
            right_interface=right_hand_can_port,
            left_urdf=left_hand_urdf,
            right_urdf=right_hand_urdf,
            control_hz=hand_control_hz,
        )

        if reset:
            print("Reset flag detected. Homing arms, opening hands, and exiting.")
            arm_controller.reset(blocking=True)
            hand_controller.reset()
            if not disable_after_reset:
                arm_controller.close(disable=False)
                arm_controller = None
                print("A7black arm motors were left enabled to hold the reset pose.")
            return

        hand_stop_event = threading.Event()
        hand_controller.start_threads(stop_event=hand_stop_event)
        arm_controller.run()
        hand_stop_event.set()

    finally:
        if hand_controller is not None:
            hand_controller.close()
        if arm_controller is not None:
            arm_controller.close(disable=disable_after_reset if reset else True)


if __name__ == "__main__":
    tyro.cli(main)

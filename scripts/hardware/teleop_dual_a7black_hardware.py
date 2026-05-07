from typing import Optional

import tyro

from xrobotoolkit_teleop.hardware.a7black_teleop_controller import (
    DEFAULT_DUAL_A7BLACK_MANIPULATOR_CONFIG,
    DEFAULT_DUAL_A7BLACK_URDF_PATH,
    DEFAULT_JOINT2_POLE_CLEARANCE_DEG,
    DualA7BlackTeleopController,
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
    hardware_map_path: str = str(DEFAULT_REALHAND_HARDWARE_MAP_PATH),
    interface_type: str = "socketcan",
    world_frame: str = "urdf",
    scale_factor: float = 1.0,
    arm_velocity: float = 1.0,
    arm_acceleration: float = 10.0,
    joint2_pole_clearance_deg: float = DEFAULT_JOINT2_POLE_CLEARANCE_DEG,
    home_timeout_sec: float = 20.0,
    home_tolerance_deg: float = 2.0,
    enable_on_start: bool = True,
    home_on_start: bool = False,
    reset: bool = False,
    disable_after_reset: bool = False,
    enable_camera: bool = False,
    enable_log_data: bool = True,
    visualize_placo: bool = True,
    control_rate_hz: int = 50,
    log_dir: str = "logs/dual_a7black",
):
    """Run dual A7black arm teleoperation on hardware.

    The arm CAN ports are intentionally separate from the RealHand L6 hand
    ports; this script currently drives only the two A7black arms.
    """
    can_ports = resolve_realhand_can_ports(
        hardware_map_path,
        required_roles=["left_arm", "right_arm"],
    )
    left_arm_can_port = left_arm_can_port or can_ports["left_arm"]
    right_arm_can_port = right_arm_can_port or can_ports["right_arm"]
    errors = validate_realhand_can_ports(
        {
            "left_arm": left_arm_can_port,
            "right_arm": right_arm_can_port,
        },
        ["left_arm", "right_arm"],
    )
    if errors:
        raise ValueError("Invalid CAN mapping:\n  - " + "\n  - ".join(errors))
    print(
        "CAN mapping: "
        f"left_arm={left_arm_can_port}, right_arm={right_arm_can_port}"
    )

    controller = DualA7BlackTeleopController(
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

    if reset:
        print("Reset flag detected. Homing A7black arms and exiting.")
        controller.reset(blocking=True)
        controller.close(disable=disable_after_reset)
        if not disable_after_reset:
            print("A7black arm motors were left enabled to hold the reset pose.")
        return

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)

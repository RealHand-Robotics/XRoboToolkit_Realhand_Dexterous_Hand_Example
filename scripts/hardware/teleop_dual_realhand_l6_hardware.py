"""Hardware teleop for dual RealHand L6 hands."""

import threading
import time
from typing import Optional

import tyro

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.realhand_hardware_map import (
    DEFAULT_REALHAND_HARDWARE_MAP_PATH,
    resolve_realhand_can_ports,
    validate_realhand_can_ports,
)
from xrobotoolkit_teleop.hardware.realhand_l6_controller import (
    DEFAULT_LEFT_URDF,
    DEFAULT_RIGHT_URDF,
    DualRealHandL6Controller,
)


def main(
    left_hand_can_port: Optional[str] = None,
    right_hand_can_port: Optional[str] = None,
    hardware_map_path: str = str(DEFAULT_REALHAND_HARDWARE_MAP_PATH),
    left_hand_urdf: str = DEFAULT_LEFT_URDF,
    right_hand_urdf: str = DEFAULT_RIGHT_URDF,
    control_hz: float = 60.0,
    reset: bool = False,
) -> None:
    """Drive dual RealHand L6 hands from XR hand tracking."""
    can_ports = resolve_realhand_can_ports(
        hardware_map_path,
        required_roles=["left_hand", "right_hand"],
    )
    left_hand_can_port = left_hand_can_port or can_ports["left_hand"]
    right_hand_can_port = right_hand_can_port or can_ports["right_hand"]
    errors = validate_realhand_can_ports(
        {
            "left_hand": left_hand_can_port,
            "right_hand": right_hand_can_port,
        },
        ["left_hand", "right_hand"],
    )
    if errors:
        raise ValueError("Invalid CAN mapping:\n  - " + "\n  - ".join(errors))
    print(
        "CAN mapping: "
        f"left_hand={left_hand_can_port}, right_hand={right_hand_can_port}"
    )

    xr_client = XrClient()
    controller = DualRealHandL6Controller(
        xr_client=xr_client,
        left_interface=left_hand_can_port,
        right_interface=right_hand_can_port,
        left_urdf=left_hand_urdf,
        right_urdf=right_hand_urdf,
        control_hz=control_hz,
    )

    if reset:
        print("Reset flag detected. Opening hands and exiting.")
        controller.reset()
        controller.close()
        return

    stop_event = threading.Event()
    controller.start_threads(stop_event=stop_event)

    try:
        while not stop_event.is_set():
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Shutting down...")
        stop_event.set()
    finally:
        controller.close()
        print("Dual RealHand L6 controller stopped.")


if __name__ == "__main__":
    tyro.cli(main)

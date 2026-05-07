#!/usr/bin/env python3
"""Bring SocketCAN interfaces up for RealHand hardware.

This script only configures Linux CAN links. It does not import the RealHand SDK
and does not communicate with the robot.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


DEFAULT_INTERFACES = ["can0", "can1", "can2", "can3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure SocketCAN links for RealHand hardware.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interfaces",
        nargs="+",
        default=DEFAULT_INTERFACES,
        help="CAN interfaces to configure.",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=1_000_000,
        help="CAN bitrate.",
    )
    parser.add_argument(
        "--force-reset",
        action="store_true",
        help="Set each interface down before configuring it. This interrupts CAN traffic.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show link details after configuring each interface.",
    )
    return parser.parse_args()


def run(command: list[str], required: bool = True) -> bool:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode == 0:
        output = completed.stdout.strip()
        if output:
            print(output)
        return True

    message = (completed.stderr or completed.stdout).strip()
    if required:
        print(f"FAIL: {' '.join(command)}", file=sys.stderr)
        if message:
            print(message, file=sys.stderr)
    return False


def configure_interface(interface: str, bitrate: int, force_reset: bool, show: bool) -> bool:
    print(f"[setup] {interface}: bitrate={bitrate}")
    if force_reset:
        run(["sudo", "ip", "link", "set", interface, "down"], required=False)

    ok = run(
        [
            "sudo",
            "ip",
            "link",
            "set",
            interface,
            "up",
            "type",
            "can",
            "bitrate",
            str(bitrate),
        ]
    )
    if ok and show:
        run(["ip", "-details", "link", "show", interface], required=False)
    return ok


def main() -> int:
    args = parse_args()
    failures = [
        interface
        for interface in args.interfaces
        if not configure_interface(interface, args.bitrate, args.force_reset, args.show)
    ]
    if failures:
        print(f"Failed to configure: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

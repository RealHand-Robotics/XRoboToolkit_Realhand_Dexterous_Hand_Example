#!/usr/bin/env python3
"""Read-only RealHand CAN hardware identification.

This script probes RealHand devices without sending motion commands. For hands it
uses version/telemetry reads. For arms it only instantiates the SDK class, whose
constructor performs the documented online motor check; it does not call
enable(), home(), move_*, set_*, save_params(), or reset_error().
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_INTERFACES = ["can0", "can1", "can2", "can3"]
DEFAULT_MODELS = ["L6"]
ALL_MODELS = ["L6", "L20Lite", "L20", "L25", "O6", "A7Lite", "A7"]
HAND_MODELS = {"L6", "L20Lite", "L20", "L25", "O6"}
ARM_MODELS = {"A7Lite", "A7"}
VALID_ROLES = {"left_hand", "right_hand", "left_arm", "right_arm"}


@dataclass
class ProbeResult:
    ok: bool
    interface: str
    model: str
    side: str
    role: str | None
    message: str
    serial_number: str | None = None
    firmware_version: str | None = None
    pcb_version: str | None = None
    mechanical_version: str | None = None
    elapsed_ms: int | None = None
    details: dict[str, Any] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify RealHand devices on CAN without moving hardware.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interfaces",
        nargs="+",
        default=DEFAULT_INTERFACES,
        help="CAN interfaces to scan.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=ALL_MODELS,
        help="RealHand SDK classes to probe. Use L6 while identifying the hand.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Probe all supported hand and arm classes.",
    )
    parser.add_argument(
        "--sides",
        nargs="+",
        default=["left", "right"],
        choices=["left", "right"],
        help="Device sides to probe.",
    )
    parser.add_argument(
        "--interface-type",
        default="socketcan",
        help="python-can interface type passed to the SDK.",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Bring interfaces up before probing using sudo ip link.",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=1_000_000,
        help="CAN bitrate used with --setup.",
    )
    parser.add_argument(
        "--force-reset-can",
        action="store_true",
        help="Set interfaces down before configuring them. This interrupts CAN traffic.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=800,
        help="Timeout for hand read-only fallback telemetry reads.",
    )
    parser.add_argument(
        "--probe-delay-sec",
        type=float,
        default=0.05,
        help="Delay between probes.",
    )
    parser.add_argument(
        "--expect",
        nargs="*",
        default=[],
        metavar="ROLE=IFACE",
        help=(
            "Expected mapping entries to verify, for example "
            "left_hand=can2 right_hand=can3. Valid roles: "
            + ", ".join(sorted(VALID_ROLES))
        ),
    )
    parser.add_argument(
        "--write-map",
        type=Path,
        help="Write confirmed mapping JSON if identification is unambiguous.",
    )
    parser.add_argument(
        "--write-log",
        type=Path,
        help="Write all probe results as JSON, including failures.",
    )
    parser.add_argument(
        "--allow-shared-interface",
        action="store_true",
        help="Allow multiple roles to be recorded on the same CAN interface.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full failure messages.",
    )
    args = parser.parse_args(argv)
    if args.all_models:
        args.models = ALL_MODELS
    args.expect = parse_expectations(args.expect)
    return args


def parse_expectations(entries: list[str]) -> dict[str, str]:
    expectations: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"Invalid --expect entry {entry!r}; use ROLE=IFACE.")
        role, interface = entry.split("=", 1)
        role = role.strip()
        interface = interface.strip()
        if role not in VALID_ROLES:
            raise SystemExit(
                f"Invalid role {role!r}; valid roles are {', '.join(sorted(VALID_ROLES))}."
            )
        if not interface:
            raise SystemExit(f"Invalid empty interface for role {role!r}.")
        if role in expectations:
            raise SystemExit(f"Duplicate expectation for {role!r}.")
        expectations[role] = interface

    reverse: dict[str, list[str]] = {}
    for role, interface in expectations.items():
        reverse.setdefault(interface, []).append(role)
    duplicates = {iface: roles for iface, roles in reverse.items() if len(roles) > 1}
    if duplicates:
        formatted = ", ".join(
            f"{iface}: {', '.join(roles)}" for iface, roles in duplicates.items()
        )
        raise SystemExit(
            "Expected map assigns more than one role to the same interface: "
            f"{formatted}. Check the CAN numbering before moving hardware."
        )
    return expectations


def setup_can_interfaces(
    interfaces: list[str], bitrate: int, force_reset: bool
) -> bool:
    ok = True
    for interface in interfaces:
        print(f"[setup] {interface}: bitrate={bitrate}")
        if force_reset:
            run_setup_command(["sudo", "ip", "link", "set", interface, "down"])
        command = [
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
        if not run_setup_command(command):
            ok = False
    return ok


def run_setup_command(command: list[str]) -> bool:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode == 0:
        return True
    message = (completed.stderr or completed.stdout).strip()
    print(f"  FAIL: {' '.join(command)}")
    if message:
        print(f"        {message}")
    return False


def load_sdk() -> Any:
    try:
        return importlib.import_module("realhand")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The realhand SDK is not installed. Install it with:\n"
            '  pip install "realhand @ git+https://github.com/RealHand-Robotics/realbot-python-sdk.git"\n'
            'For arms, use the kinetix extra:\n'
            '  pip install "realhand[kinetix] @ git+https://github.com/RealHand-Robotics/realbot-python-sdk.git"'
        ) from exc


def get_model_class(realhand: Any, model: str) -> Any:
    try:
        return getattr(realhand, model)
    except AttributeError as exc:
        raise SystemExit(
            f"The installed realhand SDK does not expose {model}. "
            "Update the SDK or remove that model from --models."
        ) from exc


def role_for(model: str, side: str) -> str:
    family = "hand" if model in HAND_MODELS else "arm"
    return f"{side}_{family}"


def probe(
    model_class: Any,
    model: str,
    side: str,
    interface: str,
    interface_type: str,
    timeout_ms: int,
) -> ProbeResult:
    started = time.monotonic()
    try:
        if model in HAND_MODELS:
            result = probe_hand(
                model_class=model_class,
                model=model,
                side=side,
                interface=interface,
                interface_type=interface_type,
                timeout_ms=timeout_ms,
            )
        elif model in ARM_MODELS:
            result = probe_arm(
                model_class=model_class,
                model=model,
                side=side,
                interface=interface,
                interface_type=interface_type,
            )
        else:
            result = ProbeResult(
                ok=False,
                interface=interface,
                model=model,
                side=side,
                role=None,
                message=f"Unsupported model family for {model}.",
            )
    except Exception as exc:
        result = ProbeResult(
            ok=False,
            interface=interface,
            model=model,
            side=side,
            role=None,
            message=format_exception(exc),
        )
    result.elapsed_ms = int((time.monotonic() - started) * 1000)
    return result


def probe_hand(
    model_class: Any,
    model: str,
    side: str,
    interface: str,
    interface_type: str,
    timeout_ms: int,
) -> ProbeResult:
    with model_class(
        side=side,
        interface_name=interface,
        interface_type=interface_type,
    ) as hand:
        stop_polling_if_available(hand)
        info_result = read_hand_version_info(hand)
        if info_result is not None:
            return ProbeResult(
                ok=True,
                interface=interface,
                model=model,
                side=side,
                role=role_for(model, side),
                message="device info read",
                **info_result,
            )

        angle_sample = read_hand_angle_sample(hand, timeout_ms)
        return ProbeResult(
            ok=True,
            interface=interface,
            model=model,
            side=side,
            role=role_for(model, side),
            message="angle telemetry read",
            details={"angles": angle_sample},
        )


def probe_arm(
    model_class: Any,
    model: str,
    side: str,
    interface: str,
    interface_type: str,
) -> ProbeResult:
    with model_class(
        side=side,
        interface_name=interface,
        interface_type=interface_type,
    ):
        return ProbeResult(
            ok=True,
            interface=interface,
            model=model,
            side=side,
            role=role_for(model, side),
            message="SDK constructor online check passed; no enable or motion command sent",
        )


def stop_polling_if_available(device: Any) -> None:
    stop_polling = getattr(device, "stop_polling", None)
    if callable(stop_polling):
        stop_polling()


def read_hand_version_info(hand: Any) -> dict[str, str | None] | None:
    version = getattr(hand, "version", None)
    if version is None or not hasattr(version, "get_device_info"):
        return None
    try:
        info = version.get_device_info()
    except Exception:
        return None
    return {
        "serial_number": stringify(getattr(info, "serial_number", None)),
        "firmware_version": stringify(getattr(info, "firmware_version", None)),
        "pcb_version": stringify(getattr(info, "pcb_version", None)),
        "mechanical_version": stringify(getattr(info, "mechanical_version", None)),
    }


def read_hand_angle_sample(hand: Any, timeout_ms: int) -> list[float]:
    angle = getattr(hand, "angle", None)
    if angle is None or not hasattr(angle, "get_blocking"):
        raise RuntimeError("No hand.version info and no hand.angle.get_blocking available.")
    data = angle.get_blocking(timeout_ms=timeout_ms)
    angles = getattr(data, "angles", data)
    return value_to_list(angles)


def stringify(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def value_to_list(value: Any) -> list[float]:
    to_list = getattr(value, "to_list", None)
    if callable(to_list):
        return [float(v) for v in to_list()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    raise RuntimeError(f"Cannot convert {type(value).__name__} to a list.")


def format_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def summarize_result(result: ProbeResult, verbose: bool) -> None:
    prefix = f"[probe] {result.interface:<5} {result.model:<7} {result.side:<5}"
    if result.ok:
        serial = f" serial={result.serial_number}" if result.serial_number else ""
        firmware = (
            f" fw={result.firmware_version}" if result.firmware_version else ""
        )
        print(f"{prefix} FOUND role={result.role}{serial}{firmware}")
        return
    message = result.message if verbose else short_message(result.message)
    print(f"{prefix} no response ({message})")


def short_message(message: str) -> str:
    if len(message) <= 100:
        return message
    return message[:97] + "..."


def build_confirmed_map(
    results: list[ProbeResult], allow_shared_interface: bool
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    successes = [result for result in results if result.ok and result.role]
    by_role: dict[str, list[ProbeResult]] = {}
    by_interface: dict[str, list[ProbeResult]] = {}
    for result in successes:
        by_role.setdefault(result.role or "", []).append(result)
        by_interface.setdefault(result.interface, []).append(result)

    for role, matches in sorted(by_role.items()):
        if len(matches) > 1:
            errors.append(
                f"{role} matched more than once: "
                + ", ".join(f"{item.model}@{item.interface}" for item in matches)
            )

    if not allow_shared_interface:
        for interface, matches in sorted(by_interface.items()):
            roles = sorted({item.role or "" for item in matches})
            if len(roles) > 1:
                errors.append(
                    f"{interface} matched multiple roles: {', '.join(roles)}"
                )

    if errors:
        return {}, errors

    identified = {
        role: result_to_map_entry(matches[0])
        for role, matches in sorted(by_role.items())
    }
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "safety": (
            "Generated by read-only identification. Motion scripts should verify "
            "role, model, side, and interface against this file before commanding hardware."
        ),
        "identified": identified,
    }, []


def result_to_map_entry(result: ProbeResult) -> dict[str, Any]:
    return {
        "model": result.model,
        "side": result.side,
        "interface": result.interface,
        "serial_number": result.serial_number,
        "firmware_version": result.firmware_version,
        "pcb_version": result.pcb_version,
        "mechanical_version": result.mechanical_version,
    }


def verify_expectations(map_data: dict[str, Any], expectations: dict[str, str]) -> list[str]:
    errors: list[str] = []
    identified = map_data.get("identified", {})
    for role, expected_interface in sorted(expectations.items()):
        entry = identified.get(role)
        if entry is None:
            errors.append(f"Expected {role} on {expected_interface}, but {role} was not found.")
            continue
        actual_interface = entry.get("interface")
        if actual_interface != expected_interface:
            errors.append(
                f"Expected {role} on {expected_interface}, but found it on {actual_interface}."
            )
    return errors


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[write] {path}")


def print_summary(results: list[ProbeResult]) -> None:
    successes = [result for result in results if result.ok]
    print()
    if not successes:
        print("No RealHand devices were identified.")
        return
    print("Identified devices:")
    for result in successes:
        serial = result.serial_number or "unknown-serial"
        print(
            f"  {result.role}: {result.model} {result.side} on "
            f"{result.interface} ({serial})"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("Read-only scan: no enable, home, move_*, set_*, save_params, or reset_error calls.")
    if args.setup:
        if not setup_can_interfaces(args.interfaces, args.bitrate, args.force_reset_can):
            return 2

    realhand = load_sdk()
    model_classes = {model: get_model_class(realhand, model) for model in args.models}
    results: list[ProbeResult] = []

    for interface in args.interfaces:
        for model in args.models:
            for side in args.sides:
                result = probe(
                    model_class=model_classes[model],
                    model=model,
                    side=side,
                    interface=interface,
                    interface_type=args.interface_type,
                    timeout_ms=args.timeout_ms,
                )
                results.append(result)
                summarize_result(result, args.verbose)
                if args.probe_delay_sec > 0:
                    time.sleep(args.probe_delay_sec)

    print_summary(results)
    if args.write_log:
        write_json(args.write_log, [asdict(result) for result in results])

    map_data, map_errors = build_confirmed_map(results, args.allow_shared_interface)
    expectation_errors = verify_expectations(map_data, args.expect) if map_data else []
    all_errors = map_errors + expectation_errors

    if all_errors:
        print("\nMap is not confirmed:")
        for error in all_errors:
            print(f"  - {error}")
        print("Do not run motion scripts until the map is corrected.")
        return 1

    if args.write_map:
        if not map_data.get("identified"):
            print("No confirmed devices to write. Do not run motion scripts.")
            return 1
        write_json(args.write_map, map_data)

    if args.expect:
        print("Expected mapping verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

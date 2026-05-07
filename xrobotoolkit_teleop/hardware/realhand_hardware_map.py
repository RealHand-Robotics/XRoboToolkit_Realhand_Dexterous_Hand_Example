"""Shared RealHand hardware role-to-CAN mapping."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path


DEFAULT_REALHAND_HARDWARE_MAP_PATH = Path(__file__).with_name("realhand_hardware_map.json")

DEFAULT_REALHAND_CAN_PORTS = {
    "left_arm": "can0",
    "right_arm": "can1",
    "left_hand": "can2",
    "right_hand": "can3",
}

EXPECTED_ROLE_SIDES = {
    "left_arm": "left",
    "right_arm": "right",
    "left_hand": "left",
    "right_hand": "right",
}

EXPECTED_ROLE_MODELS = {
    "left_arm": {"A7", "A7Lite"},
    "right_arm": {"A7", "A7Lite"},
    "left_hand": {"L6"},
    "right_hand": {"L6"},
}


def resolve_realhand_can_ports(
    map_path: str | Path | None = DEFAULT_REALHAND_HARDWARE_MAP_PATH,
    required_roles: Iterable[str] | None = None,
) -> dict[str, str]:
    """Resolve CAN ports for RealHand roles.

    If a hardware map exists, role, model, side, and interface are checked
    before returning the map values. Missing map files fall back to the lab
    default layout.
    """
    ports = dict(DEFAULT_REALHAND_CAN_PORTS)
    roles = list(required_roles or DEFAULT_REALHAND_CAN_PORTS.keys())
    map_data = _load_map(map_path)
    identified = map_data.get("identified", {}) if map_data else {}
    errors: list[str] = []

    if identified:
        for role in roles:
            entry = identified.get(role)
            if entry is None:
                errors.append(f"Hardware map is missing required role {role!r}.")
                continue
            if not isinstance(entry, Mapping):
                errors.append(f"Hardware map entry for {role!r} must be an object.")
                continue
            errors.extend(_validate_entry(role, entry))
            interface = entry.get("interface")
            if isinstance(interface, str) and interface:
                ports[role] = interface

    errors.extend(validate_realhand_can_ports(ports, roles))
    if errors:
        formatted = "\n  - ".join(errors)
        raise ValueError(f"Invalid RealHand CAN hardware map:\n  - {formatted}")

    return ports


def validate_realhand_can_ports(
    ports: Mapping[str, str],
    required_roles: Iterable[str],
) -> list[str]:
    errors = []
    selected_ports = {}
    for role in required_roles:
        interface = ports.get(role)
        if not interface:
            errors.append(f"No CAN interface configured for {role}.")
            continue
        selected_ports[role] = interface
    errors.extend(_validate_unique_interfaces(selected_ports))
    return errors


def _load_map(map_path: str | Path | None) -> dict:
    if map_path is None or str(map_path) == "":
        return {}

    path = Path(map_path).expanduser()
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Hardware map must be a JSON object: {path}")
    return data


def _validate_entry(role: str, entry: Mapping) -> list[str]:
    errors = []
    if role not in EXPECTED_ROLE_SIDES:
        errors.append(f"Unknown hardware role {role!r}.")
        return errors

    interface = entry.get("interface")
    if not isinstance(interface, str) or not interface:
        errors.append(f"{role} has no valid CAN interface.")

    expected_side = EXPECTED_ROLE_SIDES[role]
    side = entry.get("side")
    if side != expected_side:
        errors.append(f"{role} expected side {expected_side!r}, map has {side!r}.")

    model = entry.get("model")
    expected_models = EXPECTED_ROLE_MODELS[role]
    if model not in expected_models:
        allowed = ", ".join(sorted(expected_models))
        errors.append(f"{role} expected model {allowed}, map has {model!r}.")

    return errors


def _validate_unique_interfaces(ports: Mapping[str, str]) -> list[str]:
    seen: dict[str, str] = {}
    errors = []
    for role, interface in ports.items():
        previous_role = seen.get(interface)
        if previous_role is not None:
            errors.append(
                f"{interface} is assigned to both {previous_role} and {role}."
            )
        seen[interface] = role
    return errors

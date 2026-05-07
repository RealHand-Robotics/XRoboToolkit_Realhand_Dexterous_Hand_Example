"""Dual-hand MuJoCo visualization teleop for the RealHand L6 dexterous hands.

Uses XR hand tracking + dex_retargeting to drive left and right RealHand L6
URDFs in one native MuJoCo viewer. The combined MuJoCo model is generated from
the existing single-hand URDF assets at startup.
"""

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict

import mujoco
import numpy as np
import tyro
from dex_retargeting.constants import HandType, RetargetingType, RobotName
from mujoco import viewer as mj_viewer

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.dex_hand_utils import (
    DexHandTracker,
    pico_hand_state_to_mediapipe,
)
from xrobotoolkit_teleop.utils.mujoco_utils import set_mujoco_joint_pos_by_name
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def _default_urdf_dir(hand_type: str) -> str:
    return os.path.join(ASSET_PATH, "real_hand/l6", hand_type)


def _urdf_file_from_dir(urdf_dir: str, hand_type: str) -> str:
    return os.path.join(urdf_dir, f"realhand_l6_{hand_type}.urdf")


def _prefixed_urdf_elements(urdf_file: str, prefix: str) -> list[ET.Element]:
    source_path = Path(urdf_file)
    source_root = ET.parse(source_path).getroot()
    elements = []

    for link in source_root.findall("link"):
        link = ET.fromstring(ET.tostring(link))
        link.attrib["name"] = f"{prefix}_{link.attrib['name']}"
        for mesh in link.findall(".//mesh"):
            filename = mesh.attrib.get("filename")
            if filename and not Path(filename).is_absolute():
                mesh.attrib["filename"] = str((source_path.parent / filename).resolve())
        elements.append(link)

    for joint in source_root.findall("joint"):
        joint = ET.fromstring(ET.tostring(joint))
        joint.attrib["name"] = f"{prefix}_{joint.attrib['name']}"
        joint.find("parent").attrib["link"] = f"{prefix}_{joint.find('parent').attrib['link']}"
        joint.find("child").attrib["link"] = f"{prefix}_{joint.find('child').attrib['link']}"
        mimic = joint.find("mimic")
        if mimic is not None:
            mimic.attrib["joint"] = f"{prefix}_{mimic.attrib['joint']}"
        elements.append(joint)

    return elements


def _add_mount(robot: ET.Element, prefix: str, y_offset: float) -> None:
    mount = ET.SubElement(robot, "joint", {"name": f"{prefix}_mount", "type": "fixed"})
    ET.SubElement(mount, "origin", {"xyz": f"0 {y_offset} 0", "rpy": "0 0 0"})
    ET.SubElement(mount, "parent", {"link": "world"})
    ET.SubElement(mount, "child", {"link": f"{prefix}_hand_base_link"})


def build_dual_realhand_urdf(
    left_urdf_file: str,
    right_urdf_file: str,
    hand_spacing: float,
) -> str:
    robot = ET.Element("robot", {"name": "dual_realhand_l6"})
    ET.SubElement(robot, "link", {"name": "world"})

    for element in _prefixed_urdf_elements(left_urdf_file, "left"):
        robot.append(element)
    for element in _prefixed_urdf_elements(right_urdf_file, "right"):
        robot.append(element)

    _add_mount(robot, "left", hand_spacing / 2.0)
    _add_mount(robot, "right", -hand_spacing / 2.0)

    output_path = Path(tempfile.gettempdir()) / "dual_realhand_l6_mujoco.urdf"
    ET.ElementTree(robot).write(output_path, encoding="utf-8", xml_declaration=True)
    return str(output_path)


def _build_scene_model(urdf_file: str) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(urdf_file)
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.8, 0.8, 0.8]
    spec.visual.headlight.specular = [0.25, 0.25, 0.25]
    spec.visual.rgba.haze = [0.12, 0.2, 0.3, 1.0]

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_BOX
    floor.pos = [0.0, 0.0, -0.03]
    floor.size = [1.4, 1.4, 0.01]
    floor.rgba = [0.08, 0.16, 0.28, 1.0]
    floor.friction = [1.0, 0.005, 0.0001]

    grid_extent = 1.4
    grid_step = 0.2
    grid_half_width = 0.002
    grid_z = -0.018
    grid_color = [0.45, 0.65, 0.9, 1.0]
    for i, coord in enumerate(np.arange(-grid_extent, grid_extent + grid_step * 0.5, grid_step)):
        x_line = spec.worldbody.add_geom()
        x_line.name = f"grid_x_{i}"
        x_line.type = mujoco.mjtGeom.mjGEOM_BOX
        x_line.pos = [0.0, float(coord), grid_z]
        x_line.size = [grid_extent, grid_half_width, 0.001]
        x_line.rgba = grid_color

        y_line = spec.worldbody.add_geom()
        y_line.name = f"grid_y_{i}"
        y_line.type = mujoco.mjtGeom.mjGEOM_BOX
        y_line.pos = [float(coord), 0.0, grid_z]
        y_line.size = [grid_half_width, grid_extent, 0.001]
        y_line.rgba = grid_color

    light = spec.worldbody.add_light()
    light.name = "key_light"
    light.pos = [0.0, 0.0, 1.2]
    light.diffuse = [0.6, 0.6, 0.6]
    light.ambient = [0.2, 0.2, 0.2]
    light.specular = [0.15, 0.15, 0.15]
    light.dir = [0.0, 0.0, -1.0]

    return spec.compile()


def _make_tracker(urdf_dir: str, hand_type: HandType) -> DexHandTracker:
    return DexHandTracker(
        robot_name=RobotName.real,
        urdf_path=urdf_dir,
        hand_type=hand_type,
        retargeting_type=RetargetingType.vector,
    )


def _update_hand(
    xr_client: XrClient,
    mj_model: mujoco.MjModel,
    mj_qpos: np.ndarray,
    tracker: DexHandTracker,
    hand_type: str,
    prefix: str,
    joint_names: list[str],
) -> None:
    hand_state = xr_client.get_hand_tracking_state(hand_type)
    if hand_state is None:
        return

    hand_state = np.array(hand_state)
    if np.all(hand_state == 0):
        return

    mediapipe_hand_state = pico_hand_state_to_mediapipe(hand_state)
    qpos = tracker.retarget(mediapipe_hand_state)
    if qpos is None:
        return

    for joint_name, joint_pos in zip(joint_names, qpos):
        set_mujoco_joint_pos_by_name(mj_model, mj_qpos, f"{prefix}_{joint_name}", joint_pos)


def main(
    left_urdf_dir: str = "",
    right_urdf_dir: str = "",
    hand_spacing: float = 0.24,
):
    """Run the dual RealHand L6 teleoperation demo in MuJoCo.

    Args:
        left_urdf_dir: Path to the left L6 URDF directory. If omitted, this
            resolves to ``assets/real_hand/l6/left``.
        right_urdf_dir: Path to the right L6 URDF directory. If omitted, this
            resolves to ``assets/real_hand/l6/right``.
        hand_spacing: Distance between the two displayed hands in meters.
    """
    left_urdf_dir = left_urdf_dir or _default_urdf_dir("left")
    right_urdf_dir = right_urdf_dir or _default_urdf_dir("right")
    left_urdf_file = _urdf_file_from_dir(left_urdf_dir, "left")
    right_urdf_file = _urdf_file_from_dir(right_urdf_dir, "right")

    missing_paths = [path for path in [left_urdf_file, right_urdf_file] if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError(f"RealHand L6 URDF not found: {missing_paths[0]}")

    dual_urdf_file = build_dual_realhand_urdf(left_urdf_file, right_urdf_file, hand_spacing)
    mj_model = _build_scene_model(dual_urdf_file)
    mj_data = mujoco.MjData(mj_model)

    xr_client = XrClient()
    trackers: Dict[str, DexHandTracker] = {
        "left": _make_tracker(left_urdf_dir, HandType.left),
        "right": _make_tracker(right_urdf_dir, HandType.right),
    }
    joint_names = {
        "left": trackers["left"].retargeting.joint_names,
        "right": trackers["right"].retargeting.joint_names,
    }

    with mj_viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -30
        viewer.cam.distance = 0.55
        viewer.cam.lookat = [0.0, 0.0, 0.08]

        while True:
            qpos = mj_data.qpos.copy()
            _update_hand(xr_client, mj_model, qpos, trackers["left"], "left", "left", joint_names["left"])
            _update_hand(xr_client, mj_model, qpos, trackers["right"], "right", "right", joint_names["right"])

            mj_data.qpos[:] = qpos
            mj_data.qvel[:] = 0.0
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()


if __name__ == "__main__":
    tyro.cli(main)

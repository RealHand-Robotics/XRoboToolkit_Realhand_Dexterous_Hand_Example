"""MuJoCo visualization teleop for the RealHand L6 dexterous hand.

Uses XR hand tracking + dex_retargeting to drive the RealHand L6 URDF in the
native MuJoCo viewer. This is a simulation-only script; no CAN bus or hand
hardware is required.
"""

import os

import mujoco
import numpy as np
import pinocchio as pin
import tyro
from dex_retargeting.constants import HandType, RetargetingType, RobotName
from mujoco import viewer as mj_viewer

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.dex_hand_utils import (
    DexHandTracker,
    pico_hand_state_to_mediapipe,
)
from xrobotoolkit_teleop.utils.mujoco_utils import calc_mujoco_qpos_from_pin_q
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def _default_urdf_dir(hand_type: str) -> str:
    return os.path.join(ASSET_PATH, "real_hand/l6", hand_type)


def _urdf_file_from_dir(urdf_dir: str, hand_type: str) -> str:
    return os.path.join(urdf_dir, f"realhand_l6_{hand_type}.urdf")


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
    floor.size = [1.2, 1.2, 0.01]
    floor.rgba = [0.08, 0.16, 0.28, 1.0]
    floor.friction = [1.0, 0.005, 0.0001]

    grid_extent = 1.2
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


def main(
    urdf_dir: str = "",
    hand_type: str = "right",
):
    """Run the RealHand L6 teleoperation demo in MuJoCo.

    Args:
        urdf_dir: Path to the L6 URDF directory. If omitted, this resolves to
            ``assets/real_hand/l6/<hand_type>``. The script loads the
            ``realhand_l6_<hand_type>.urdf`` file inside this directory.
        hand_type: ``"left"`` or ``"right"``.
    """
    if hand_type not in ["left", "right"]:
        raise ValueError("hand_type must be 'left' or 'right'")

    urdf_dir = urdf_dir or _default_urdf_dir(hand_type)
    urdf_file = _urdf_file_from_dir(urdf_dir, hand_type)
    if not os.path.exists(urdf_file):
        raise FileNotFoundError(f"RealHand L6 URDF not found: {urdf_file}")

    mj_model = _build_scene_model(urdf_file)
    mj_data = mujoco.MjData(mj_model)
    pin_model = pin.buildModelFromUrdf(urdf_file)

    xr_client = XrClient()

    dex_hand_type = HandType.left if hand_type == "left" else HandType.right
    dextracker = DexHandTracker(
        robot_name=RobotName.real,
        urdf_path=urdf_dir,
        hand_type=dex_hand_type,
        retargeting_type=RetargetingType.vector,
    )

    with mj_viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -30
        viewer.cam.distance = 0.45
        viewer.cam.lookat = [0.0, 0.0, 0.08]

        while True:
            hand_state = xr_client.get_hand_tracking_state(hand_type)
            if hand_state is None:
                continue

            hand_state = np.array(hand_state)
            if np.all(hand_state == 0):
                print("all zero, ignore")
                continue

            mediapipe_hand_state = pico_hand_state_to_mediapipe(hand_state)
            pin_q = dextracker.retarget(mediapipe_hand_state)
            if pin_q is None:
                continue

            mj_data.qpos[:] = calc_mujoco_qpos_from_pin_q(mj_model, pin_model, pin_q)
            mj_data.qvel[:] = 0.0
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()


if __name__ == "__main__":
    tyro.cli(main)

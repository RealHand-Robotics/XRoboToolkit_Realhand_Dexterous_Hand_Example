"""Dual-hand Placo visualization teleop for the RealHand L6 dexterous hands.

Uses XR hand tracking + dex_retargeting to drive left and right RealHand L6
URDFs in one browser-based Placo viewer. This is a simulation-only script; no
CAN bus or hand hardware is required.
"""

import os
import webbrowser

import numpy as np
import placo
import tyro
from dex_retargeting.constants import HandType, RetargetingType, RobotName
from placo_utils.visualization import robot_viz

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.dex_hand_utils import (
    DexHandTracker,
    pico_hand_state_to_mediapipe,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def _make_tracker(urdf_path: str, hand_type: HandType) -> DexHandTracker:
    return DexHandTracker(
        robot_name=RobotName.real,
        urdf_path=urdf_path,
        hand_type=hand_type,
        retargeting_type=RetargetingType.vector,
    )


def _update_hand(
    xr_client: XrClient,
    robot: placo.RobotWrapper,
    tracker: DexHandTracker,
    hand_type: str,
) -> None:
    hand_state = xr_client.get_hand_tracking_state(hand_type)
    if hand_state is None:
        return

    hand_state = np.array(hand_state)
    if np.all(hand_state == 0):
        return

    mediapipe_hand_state = pico_hand_state_to_mediapipe(hand_state)
    qpos = tracker.retarget(mediapipe_hand_state)
    if qpos is not None:
        robot.state.q[7:] = qpos


def main(
    left_urdf_path: str = os.path.join(ASSET_PATH, "real_hand/l6/left"),
    right_urdf_path: str = os.path.join(ASSET_PATH, "real_hand/l6/right"),
    hand_spacing: float = 0.24,
):
    """Run the dual RealHand L6 teleoperation demo in Placo.

    Args:
        left_urdf_path: Path to the left L6 URDF directory.
        right_urdf_path: Path to the right L6 URDF directory.
        hand_spacing: Distance between the two displayed hands in meters.
    """
    left_robot = placo.RobotWrapper(left_urdf_path)
    right_robot = placo.RobotWrapper(right_urdf_path)

    left_robot.state.q[:3] = [0.0, hand_spacing / 2.0, 0.0]
    right_robot.state.q[:3] = [0.0, -hand_spacing / 2.0, 0.0]

    left_viz = robot_viz(left_robot, "realhand_l6_left")
    right_viz = robot_viz(right_robot, "realhand_l6_right")
    webbrowser.open(left_viz.viewer.url())

    xr_client = XrClient()
    left_tracker = _make_tracker(left_urdf_path, HandType.left)
    right_tracker = _make_tracker(right_urdf_path, HandType.right)

    while True:
        _update_hand(xr_client, left_robot, left_tracker, "left")
        _update_hand(xr_client, right_robot, right_tracker, "right")

        left_viz.display(left_robot.state.q)
        right_viz.display(right_robot.state.q)


if __name__ == "__main__":
    tyro.cli(main)

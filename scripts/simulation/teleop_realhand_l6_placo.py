"""Placo visualization teleop for the RealHand L6 dexterous hand.

Uses XR hand tracking + dex_retargeting to drive the RealHand L6 URDF in a
browser-based Placo viewer. This is a simulation-only script; no CAN bus or
hand hardware is required.
"""

import os
import webbrowser
from typing import List

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


def retarget_qpos_to_hand_angles(qpos) -> List[float]:
    """Map dex_retargeting 11-dof output to the 6 RealHand L6 finger angles
    expected by the new realhand SDK (0-100 range, here returned in radians
    for visualization)."""
    if len(qpos) != 11:
        print(
            f"unexpected retargeting qpos length {len(qpos)}, expected 11 for L6"
        )
        return [-1.0] * 6
    return [
        np.pi / 2 - (qpos[9] + qpos[10]) / 2,  # pinky
        np.pi / 2 - qpos[8],                   # ring/thumb abd depending on URDF
        np.pi / 2 - qpos[0],                   # index
        np.pi / 2 - qpos[3],                   # middle
        np.pi / 2 - qpos[5],                   # ring
        np.pi / 2 - qpos[6],                   # thumb flex
    ]


def main(
    urdf_path: str = os.path.join(ASSET_PATH, "real_hand/l6/right"),
    hand_type: str = "right",
):
    """Run the RealHand L6 teleoperation demo in Placo.

    Args:
        urdf_path: Path to the L6 URDF *directory* (e.g.
            ``assets/real_hand/l6/right``). Placo auto-loads ``robot.urdf``
            from that directory; ``DexHandTracker`` uses the directory's
            grandparent (``assets/real_hand``) as the retargeting base dir
            so it matches the yaml ``urdf_path: l6/right/...``.
        hand_type: ``"left"`` or ``"right"``.
    """
    if hand_type not in ["left", "right"]:
        raise ValueError("hand_type must be 'left' or 'right'")

    robot = placo.RobotWrapper(urdf_path)
    viz = robot_viz(robot, "RealHand L6")
    webbrowser.open(viz.viewer.url())

    xr_client = XrClient()

    dex_hand_type = HandType.left if hand_type == "left" else HandType.right
    dextracker = DexHandTracker(
        robot_name=RobotName.real,
        urdf_path=urdf_path,
        hand_type=dex_hand_type,
        retargeting_type=RetargetingType.vector,
    )

    while True:
        viz.display(robot.state.q)
        hand_state = xr_client.get_hand_tracking_state(hand_type)

        if hand_state is None:
            continue

        hand_state = np.array(hand_state)
        if np.all(hand_state == 0):
            print("all zero, ignore")
            continue

        mediapipe_hand_state = pico_hand_state_to_mediapipe(hand_state)
        qpos = dextracker.retarget(mediapipe_hand_state)
        if qpos is None:
            continue

        finger_angles = retarget_qpos_to_hand_angles(qpos)
        print(np.around(finger_angles, 2))

        robot.state.q[7:] = qpos


if __name__ == "__main__":
    tyro.cli(main)

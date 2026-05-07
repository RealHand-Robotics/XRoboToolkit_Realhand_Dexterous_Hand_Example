import os

import numpy as np
import tyro

from xrobotoolkit_teleop.simulation.placo_teleop_controller import (
    PlacoTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


A7_JOINT_NAMES = [f"{side}_joint{i}" for side in ("left", "right") for i in range(1, 8)]
DEFAULT_JOINT2_POLE_CLEARANCE_DEG = 10.0
A7_READY_JOINTS_RAD = {
    "left_joint4": 0.3490658503988659,
    "right_joint4": 0.3490658503988659,
}


def _a7_ready_joints(clearance_deg: float) -> dict[str, float]:
    ready_joints = dict(A7_READY_JOINTS_RAD)
    if clearance_deg > 0.0:
        clearance = np.deg2rad(float(clearance_deg))
        ready_joints["left_joint2"] = clearance
        ready_joints["right_joint2"] = -clearance
    return ready_joints


def main(
    robot_urdf_path: str = os.path.join(ASSET_PATH, "real_hand/A7black/dual_A7black.urdf"),
    scale_factor: float = 1.0,
    joint2_pole_clearance_deg: float = DEFAULT_JOINT2_POLE_CLEARANCE_DEG,
):
    """Run dual A7black arm teleoperation with Placo visualization."""
    config = {
        "left_arm": {
            "link_name": "left_link8",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
        },
        "right_arm": {
            "link_name": "right_link8",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
        },
    }
    ready_joints = _a7_ready_joints(joint2_pole_clearance_deg)

    controller = PlacoTeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        q_init=np.array([ready_joints.get(joint, 0.0) for joint in A7_JOINT_NAMES]),
    )

    if joint2_pole_clearance_deg > 0.0:
        clearance = np.deg2rad(joint2_pole_clearance_deg)
        controller.placo_robot.set_joint_limits("left_joint2", clearance, 3.2)
        controller.placo_robot.set_joint_limits("right_joint2", -3.2, -clearance)

    joints_task = controller.solver.add_joints_task()
    joints_task.set_joints(
        {
            joint: ready_joints.get(joint, 0.0)
            for joint in A7_JOINT_NAMES
        }
    )
    joints_task.configure("joints_regularization", "soft", 1e-4)
    controller.solver.enable_velocity_limits(True)

    print("Starting dual A7black Placo teleoperation...")
    print("Control mapping:")
    print("  - Left controller grip -> left_link8")
    print("  - Right controller grip -> right_link8")

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)

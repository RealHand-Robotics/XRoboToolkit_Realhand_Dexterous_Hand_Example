# Detailed descriptions on the python teleoperation examples

## XR client 
- https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python/blob/main/xrobotoolkit_teleop/utils/xr_client.py
- connects to XR device using the python binding of [xrobotoolkit-sdk](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind)

## Placo
- [PlaCo](https://placo.readthedocs.io/en/latest/) is an implementation for whole-body inverse kinematics and dynamics based on Quadratic Programming solver.
- Tasks: following are the main placo tasks used in this repo
  - Frame Task
    - Constraints position and orientation of a frame, used to define end-effector tracking goals for IK
  - Manipulability Task
    - Makes robot motion stable when commanded position is close to singularity or outside of workspace
    - Based on this paper: https://arxiv.org/abs/2002.11901
  - Kinetic Energy Regularization
    - Minimizes the kinetic energy of the system
  - Joint Task
    - Regularizes the joint state to be close to the default state

## Mujoco simulation
- Robot definition files: both `.xml` and `.urdf` files are required and they should be consistent with each other (same link names and joint names). The `.xml` file is for mujoco simulation, and the `.urdf` is for placo. Optionally, there should be 1 additional free floating body per end effector defined in the `.xml` file for visualization of commanded teleop targets in mujoco.
- Teleoperation task is defined by a config dict
  - link_name: the name of end effector link as defined in mujoco .xml & .urdf files
  - pose_source: name of the source of pose to be used by XrClient (e.g., `left_controller`, `right_controller`)
  - control_trigger: the key to define whether an arm control is active
  - control_mode: optional field to specify tracking mode - "pose" (default, full 6DOF) or "position" (3DOF position only)
  - vis_target: name of the body for teleop target visualization
  - motion_tracker: optional config for additional motion tracker to control another link in the manipulator (not recommended for 6DOF arms like UR5)
    - serial: serial number of the motion tracker device
    - link_target: name of the robot link to be controlled by the motion tracker
    ```python
    config = {
        "right_hand": {
            "link_name": "flange",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "control_mode": "position", # optional: "pose" (default) or "position"
            "motion_tracker": {
                "serial": "PC2310BLH9020740B",
                "link_target": "link4",
            },
            "vis_target": "right_target", # optional, only used in mujoco
        },
        "left_hand": {
            "link_name": "left_tool0",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "gripper_trigger": "left_trigger",
            "vis_target": "left_target", # optional, only used in mujoco
        },
    }
    ```

- Run mujoco demo for dual UR5e with the following script
    ```bash
    python scripts/simulation/teleop_dual_ur5e_mujoco.py
    ```

- Controlling parallel gripper in mujoco simulation
  - Users can add an optional gripper configuration in the end effector config dict
    - joint_name: the actuated mujoco joint within the gripper
    - gripper_trigger: name of the key mapped to this gripper from the controller
    - open_pos: the value of the actuated joint when fully opened
    - close_pos: the value of the actuated joint when fully closed
    ```python
    config = {
        "right_hand": {
            # other configs,
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "right_trigger",
                "joint_names": ["right_gripper_finger_joint1",],
                "open_pos": [0.05,],
                "close_pos": [0.0,],
            },
        },
        "left_hand": {
            # other configs,
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "left_trigger",
                "joint_names": ["left_gripper_finger_joint1",],
                "open_pos": [0.05,],
                "close_pos": [0.0,],
            },
        },
    }
    ```
  - Note that the parallel gripper might contain multiple joints in the `.xml` file, but only 1 of the joints should be actuated, the others should be controlled by additional equality constraints in the xml. The `.urdf` file supplied to Placo does not have to contain the gripper dof.
    ```xml
    <equality>
    <joint name="right_gripper_constraint" joint1="right_gripper_finger_joint1" joint2="right_gripper_finger_joint2" polycoef="0 -1 0 0 0" />
    <joint name="left_gripper_constraint" joint1="left_gripper_finger_joint1" joint2="left_gripper_finger_joint2" polycoef="0 -1 0 0 0" />
    </equality>
    ```
- Example of mujoco teleoperation with gripper control using dual A1X arm
    ```bash
    python scripts/simulation/teleop_dual_a1x_mujoco.py
    ```

## UR5 Hardware teleoperation:
- Robot definition files: only `.urdf` file is required
- Config
  - link_name: the name of end effector link as defined in mujoco .xml & .urdf files
  - pose_source: name of the source of pose to be used by XrClient (e.g., `left_controller`, `right_controller`)
  - control_trigger: the key to define whether an arm control is active
  - control_mode: optional field to specify tracking mode - "pose" (default, full 6DOF) or "position" (3DOF position only)
  - gripper_trigger: name of the key mapped to this gripper from the controller
    ```python  
    DEFAULT_END_EFFECTOR_CONFIG = {
        "left_arm": {
            "link_name": "left_tool0",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "gripper_trigger": "left_trigger",
        },
        "right_arm": {
            "link_name": "right_tool0",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "gripper_trigger": "right_trigger",
        },
    }
    ```
- Example code is provided for a lab setup with dual ur5e manipulators with robotiq 2f85 grippers, as well as a 2 DOF head controlled by 2 dynamixel motors.
    ```bash
    python scripts/hardware/teleop_dual_ur5e_hardware.py
    ```

# ARX R5 Hardware teleoperation
- Robot definition files: only `.urdf` is needed.
- Config: Dual arm configuration with gripper support.
  ```python
    DEFAULT_DUAL_ARX_R5_MANIPULATOR_CONFIG = {
        "right_arm": {
            "link_name": "right_link6",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "right_trigger",
                "joint_names": ["right_joint7"],
                "open_pos": [4.9],
                "close_pos": [0.0],
            },
        },
        "left_arm": {
            "link_name": "left_link6",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "left_trigger",
                "joint_names": ["left_joint7"],
                "open_pos": [4.9],
                "close_pos": [0.0],
            },
        },
    }
  ```
- run the following script
  ```bash
  python scripts/hardware/teleop_dual_arx_r5_hardware.py
  ```
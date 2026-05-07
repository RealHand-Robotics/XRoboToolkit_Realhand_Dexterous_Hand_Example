# XRoboToolkit RealHand Dexterous Hand Example

Pico XR teleoperation demo, in Python, for the **RealHand L6** dexterous hand
in Placo visualization and on real hardware. A few additional arm-teleop demos
(UR5e, ARX R5, Galaxea R1 Lite) are kept from the parent project for reference.

## Overview

This project drives the RealHand L6 hand from XR (VR/AR) controller and hand
tracking input. Hand tracking is retargeted to the L6 joint space with
[`dex_retargeting`](https://github.com/dexsuite/dex-retargeting); commands are
sent to the hardware through the new
[`realhand`](https://github.com/RealHand-Robotics/realbot-python-sdk) Python SDK
(CAN-bus based).

## Installation
1. Download and install [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service). Run the installed program before running the following demo.

2.  **Clone the repository:**
    ```bash
    git clone <this-repo-url>
    cd XRoboToolkit_Realhand_Dexterous_Hand_Example
    ```

3.  **Installation**
    **Note:** The setup scripts are currently tested on Ubuntu 22.04 / 24.04.
    It is recommended to set up a Conda environment and install using the included script.
    ```bash
    bash setup_conda.sh --conda <optional_env_name>
    conda activate <env_name>
    bash setup_conda.sh --install
    ```

    If installing on system python:
    ```bash
    bash setup.sh
    ```

    The install scripts automatically:
    - Clone and build [`XRoboToolkit-PC-Service-Pybind`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind) into `dependencies/`.
    - `pip install git+https://github.com/SherwinDengxiong/dex_retargeting_local.git`
      — the RealHand fork of `dex_retargeting` that adds the `real` robot entry
      and the RealHand YAML configs required here. The PyPI release does
      **not** have these entries, so do not replace this with
      `pip install dex_retargeting`.
    - `pip install git+https://github.com/RealHand-Robotics/realbot-python-sdk.git`
      — the new RealHand Python SDK.
    - `pip install -e .` the teleop package itself.

## Usage
Use the following instructions to run example scripts. For a more detailed description, please refer to [`teleop_details.md`](teleop_details.md).



- **RealHand L6 in Placo visualization** (the main demo for this project)
    ```bash
    python scripts/simulation/teleop_realhand_l6_placo.py --hand-type right
    ```
    This loads the L6 URDF from `assets/real_hand/l6/`, builds a dex_retargeting
    optimizer for the `real` robot, and drives the L6 in a browser-based Placo
    viewer from live XR hand tracking.

- **RealHand L6 in MuJoCo visualization**
    ```bash
    python scripts/simulation/teleop_realhand_l6_mujoco.py --hand-type right
    ```
    This loads the same RealHand L6 URDF directly in MuJoCo and drives the
    simulated hand from live XR hand tracking.

- **Dual RealHand L6 in Placo visualization**
    ```bash
    python scripts/simulation/teleop_dual_realhand_l6_placo.py
    ```

- **Dual RealHand L6 in MuJoCo visualization**
    ```bash
    python scripts/simulation/teleop_dual_realhand_l6_mujoco.py
    ```

- **Dual UR5e + Dual RealHand L6 in MuJoCo visualization**
    ```bash
    python scripts/simulation/teleop_dual_ur5e_realhand_l6_mujoco.py
    ```
    This combines the dual-UR5e arm teleop with RealHand L6 control. The grip
    button keeps the existing UR5e arm-follow behavior when controller teleop
    is active. In hand-tracking mode, the tracked palm root drives the UR5e
    end effector position while keeping its current orientation, and the
    tracked finger joints drive the RealHand L6. Palm-driven arm motion is
    intentionally scaled down for gentler teleoperation. When hand tracking is
    unavailable on a side, the trigger becomes the fallback open/close hand
    control for that side.

- **Dual A7black + Dual RealHand L6 in MuJoCo visualization**
    ```bash
    python scripts/simulation/teleop_dual_a7black_realhand_l6_mujoco.py
    ```
    This generates a MuJoCo scene from `assets/real_hand/A7black/dual_A7black.urdf`
    and attaches left/right RealHand L6 hands. Grip/controller teleop drives
    the A7black arms, hand tracking drives the L6 fingers when available, and
    trigger input falls back to open/close hand control.

### Running RealHand L6 Hardware Demo

To drive a physical RealHand L6 via the new `realhand` SDK:

```bash
# Bring up the hand CAN interfaces (one-time setup before each run):
python xrobotoolkit_teleop/hardware/setup_can_interfaces.py --interfaces can2 can3

# Teleoperate both L6 hands with the default mapping: left=can2, right=can3.
python scripts/hardware/teleop_dual_realhand_l6_hardware.py

# Send an "open hand and exit" command (useful as a safe reset):
python scripts/hardware/teleop_dual_realhand_l6_hardware.py --reset
```
This script initializes [`RealHandL6Controller`](xrobotoolkit_teleop/hardware/realhand_l6_controller.py),
which uses the new realhand SDK's `L6(side=..., interface_name=...)` interface
and sends `angle.set_angles(...)` commands on the 0-100 scale, where `100` is
fully open.





## Dependencies
XR Robotics dependencies:
- [`xrobotoolkit_sdk`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind): Python binding for XRoboToolkit PC Service SDK, MIT License

Robotics Simulation and Solver
- [`mujoco`](https://github.com/google-deepmind/mujoco): robotics simulation, Apache 2.0 License
- [`placo`](https://github.com/rhoban/placo): inverse kinematics, MIT License
- [`dex_retargeting`](https://github.com/dexsuite/dex-retargeting): XR hand → dexterous-hand retargeting. Vendored under `dependencies/dex_retargeting_local/` with custom `real` robot entries and RealHand L6/L20 configs.

Hardware Control
- [`realhand`](https://github.com/RealHand-Robotics/realbot-python-sdk): Python SDK for the RealHand L6 / L20 / L25 / O6 hands and A7 / A7 Lite arms. Installed from git by `setup.sh` / `setup_conda.sh`.
- [`dynamixel_sdk`](https://github.com/ROBOTIS-GIT/DynamixelSDK.git): Dynamixel control functions, Apache-2.0 License
- [`ur_rtde`](https://gitlab.com/sdurobotics/ur_rtde): interface for controlling and receiving data from a UR robot, MIT License


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

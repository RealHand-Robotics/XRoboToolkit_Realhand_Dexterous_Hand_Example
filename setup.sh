#!/bin/bash
set -e

# Check the operating system
OS_NAME=$(uname -s)
OS_VERSION=""

if [[ "$OS_NAME" == "Linux" ]]; then
    if command -v lsb_release &>/dev/null; then
        OS_VERSION=$(lsb_release -rs)
    elif [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS_VERSION=$VERSION_ID
    fi
    if [[ "$OS_VERSION" != "22.04" && "$OS_VERSION" != "24.04" ]]; then
        echo "Warning: This script has only been tested on Ubuntu 22.04 and 24.04"
        echo "Your system is running Ubuntu $OS_VERSION."
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation cancelled."
            exit 1
        fi
    fi
else
    echo "Unsupported operating system: $OS_NAME"
    exit 1
fi

echo "Operating system check passed: $OS_NAME $OS_VERSION"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DEX_RETARGETING_REPO_URL="${DEX_RETARGETING_REPO_URL:-git+https://github.com/SherwinDengxiong/dex_retargeting_local.git}"

# ---------------------------------------------------------------------
# External dependencies
# ---------------------------------------------------------------------
mkdir -p dependencies
cd dependencies

# XRoboToolkit SDK (VR/AR interface)
if [ ! -d "XRoboToolkit-PC-Service-Pybind" ]; then
    git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git
fi
(cd XRoboToolkit-PC-Service-Pybind && bash setup_ubuntu.sh)

# ARX R5 SDK (optional, kept for dual-arm teleop scripts)
if [ ! -d "R5" ]; then
    git clone https://github.com/zhigenzhao/R5.git
    (cd R5 && git checkout dev/python_pkg)
fi
python -m pip install ./R5/py/ARX_R5_python

cd "$REPO_ROOT"

# ---------------------------------------------------------------------
# dex_retargeting (RealHand fork hosted on GitHub)
# ---------------------------------------------------------------------
python -m pip install --force-reinstall "$DEX_RETARGETING_REPO_URL"

# ---------------------------------------------------------------------
# RealHand SDK (new realbot-python-sdk)
# ---------------------------------------------------------------------
python -m pip install git+https://github.com/RealHand-Robotics/realbot-python-sdk.git

# ---------------------------------------------------------------------
# Finally, install this project itself
# ---------------------------------------------------------------------
python -m pip install -e . || { echo "Failed to install xrobotoolkit_teleop with pip"; exit 1; }

echo -e "\n[INFO] xrobotoolkit_teleop (RealHand example) installed successfully.\n"

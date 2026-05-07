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
DEX_RETARGETING_REPO_URL="${DEX_RETARGETING_REPO_URL:-git+https://github.com/SherwinDengxiong/dex_retargeting_local.git}"

# ---------------------------------------------------------------------
# Mode 1: create the conda environment
# ---------------------------------------------------------------------
if [[ "$1" == "--conda" ]]; then
    if [[ -n "$2" ]]; then
        ENV_NAME="$2"
    else
        ENV_NAME="xr-robotics"
    fi

    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1)
    elif command -v python &>/dev/null; then
        PYTHON_VERSION=$(python --version 2>&1)
    else
        echo "Python is not installed on this system."
        exit 1
    fi

    echo "The system's default Python version is: $PYTHON_VERSION"
    PYTHON_MAJOR_MINOR=$(echo $PYTHON_VERSION | grep -oP '\d+\.\d+')

    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Conda initialization script not found. Please install Miniconda or Anaconda."
        exit 1
    fi

    conda deactivate || true
    conda remove -n "$ENV_NAME" --all -y || true
    conda create -n "$ENV_NAME" python=$PYTHON_MAJOR_MINOR -y

    echo "Conda environment '$ENV_NAME' created with Python $PYTHON_MAJOR_MINOR"
    conda activate "$ENV_NAME"
    conda deactivate

    echo -e "[INFO] Created conda environment '$ENV_NAME'.\n"
    echo -e "\t1. Activate:    conda activate $ENV_NAME"
    echo -e "\t2. Install:     bash setup_conda.sh --install"
    echo -e "\t3. Deactivate:  conda deactivate\n"

# ---------------------------------------------------------------------
# Mode 2: install packages into the active conda environment
# ---------------------------------------------------------------------
elif [[ "$1" == "--install" ]]; then
    if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
        echo "Error: No conda environment is currently activated."
        echo "Please activate a conda environment first with: conda activate <env_name>"
        exit 1
    fi
    ENV_NAME=${CONDA_DEFAULT_ENV}

    if [[ "$OS_NAME" == "Linux" ]]; then
        conda install -c conda-forge libstdcxx-ng -y
    fi
    python -m pip install uv
    python -m uv pip install --upgrade pip

    UV="python -m uv"

    cd "$REPO_ROOT"
    mkdir -p dependencies
    cd dependencies

    # XRoboToolkit SDK (VR/AR interface)
    if [ ! -d "XRoboToolkit-PC-Service-Pybind" ]; then
        git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git
    fi
    (cd XRoboToolkit-PC-Service-Pybind && bash setup_ubuntu.sh)

    # ARX R5 SDK (optional, used by dual-arm teleop scripts)
    if [ ! -d "R5" ]; then
        git clone https://github.com/zhigenzhao/R5.git
        (cd R5 && git checkout dev/python_pkg)
    fi
    $UV pip install ./R5/py/ARX_R5_python

    cd "$REPO_ROOT"

    # dex_retargeting (RealHand fork hosted on GitHub)
    $UV pip install --force-reinstall "$DEX_RETARGETING_REPO_URL"

    # RealHand SDK (new realbot-python-sdk)
    $UV pip install git+https://github.com/RealHand-Robotics/realbot-python-sdk.git

    # Finally, this project
    $UV pip install -e . || { echo "Failed to install xrobotoolkit_teleop with pip"; exit 1; }

    echo -e "\n[INFO] xrobotoolkit_teleop (RealHand example) installed in conda environment '$ENV_NAME'.\n"
else
    echo "Invalid argument. Use --conda to create a conda environment or --install to install the package."
    exit 1
fi

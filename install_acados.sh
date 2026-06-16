#!/bin/bash
# -----------------------------------------------------------------------------
# Script: install_acados_venv.sh
# Description: Sets up a Python virtual environment with Acados (optimal control)
#              on Ubuntu. Builds Acados from source and installs the Python
#              bindings (acados_template) into the venv.
#
# Usage: ./install_acados_venv.sh [VENV_DIR]
#        If VENV_DIR is not provided, defaults to ./venv_acados
# -----------------------------------------------------------------------------

set -e  # Exit on any error

# ---------------------------- Configuration ----------------------------------
VENV_DIR="/workspaces/ros2_exp/venv_acados"          # Virtual environment directory
ACADOS_SRC_DIR="/workspaces/ros2_exp/acados" # Where to clone/build Acados
ACADOS_BUILD_DIR="$ACADOS_SRC_DIR/build"
ACADOS_INSTALL_DIR="$ACADOS_SRC_DIR"    # Install to source root (typical)

# Python package requirements (beyond what Acados installs)
PIP_BUILD_REQUIRES="setuptools wheel Cython"
PIP_REQUIRES="numpy casadi PyYAML typing_extensions catkin_pkg empy==3.3.4 lark"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ---------------------------- Helper Functions -------------------------------
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        error "$1 not found. Please install it first."
    fi
}

# ---------------------------- System Dependencies ----------------------------
info "Updating package list and installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    python3-venv \
    python3-pip \
    python3-dev

# ---------------------------- Virtual Environment ----------------------------
info "Creating Python virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"
# Prevent colcon from scanning setup.py examples inside site-packages.
touch "$VENV_DIR/COLCON_IGNORE"

# Activate venv
source "$VENV_DIR/bin/activate"

info "Upgrading pip and installing Python build prerequisites"
pip install --upgrade pip
pip install --upgrade $PIP_BUILD_REQUIRES

info "Installing Python runtime prerequisites"
pip install $PIP_REQUIRES

# ---------------------------- Clone Acados ------------------------------------
if [ -d "$ACADOS_SRC_DIR" ]; then
    info "Acados source directory already exists. Skipping clone."
else
    info "Cloning Acados repository..."
    git clone https://github.com/acados/acados.git "$ACADOS_SRC_DIR"
fi

cd $ACADOS_SRC_DIR
git submodule update --recursive --init

# ---------------------------- Build Acados ------------------------------------
info "Building Acados (CMake + make)..."
mkdir -p "$ACADOS_BUILD_DIR"
cd "$ACADOS_BUILD_DIR"

# Configure with BLAS/LAPACK and shared libraries
cmake -DCMAKE_INSTALL_PREFIX="$ACADOS_INSTALL_DIR" \
      -DBUILD_SHARED_LIBS=ON \
      -DACADOS_WITH_HPIPM=ON \
      ..

make -j$(nproc)
make install

# Return to original directory
cd - > /dev/null

# ---------------------------- Set Environment Variables -----------------------
info "Setting up environment variables for Acados"
# These will be used when running Python code. We'll create an activation script
# that sources the venv and also sets the necessary paths.

ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
# Append Acados environment variables to the venv's activate script (if not already there)
if ! grep -q "ACADOS_SOURCE_DIR" "$ACTIVATE_SCRIPT"; then
    cat <<EOF >> "$ACTIVATE_SCRIPT"

# Acados environment variables
export ACADOS_SOURCE_DIR="$ACADOS_SRC_DIR"
export LD_LIBRARY_PATH="$ACADOS_INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
EOF
    info "Added Acados environment variables to $ACTIVATE_SCRIPT"
else
    info "Acados environment variables already present in $ACTIVATE_SCRIPT"
fi

# Also set for current shell
export ACADOS_SOURCE_DIR="$ACADOS_SRC_DIR"
export LD_LIBRARY_PATH="$ACADOS_INSTALL_DIR/lib:$LD_LIBRARY_PATH"

# ---------------------------- Install Python Interface -----------------------
info "Installing acados_template Python package into venv"
cd "$ACADOS_SRC_DIR/interfaces/acados_template"
pip install -e .
cd - > /dev/null

# ---------------------------- Verification ------------------------------------
info "Verifying installation..."
python -c "from acados_template import AcadosModel; print('Acados Python interface imported successfully.')"

# ---------------------------- Final Message -----------------------------------
info "Installation completed successfully!"
info "To activate the virtual environment with Acados, run:"
echo "    source $VENV_DIR/bin/activate"
info "After activation, your environment will have ACADOS_SOURCE_DIR and LD_LIBRARY_PATH set."
info "You can now use Acados in Python."

# Deactivate the virtual environment (just for cleanliness)
deactivate

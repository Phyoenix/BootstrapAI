#!/bin/bash
# Setup script for Neural Rendering Research Environment

set -e

echo "=== Setting up Neural Rendering Research Environment ==="

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA not found. Installing CUDA toolkit..."
    # Note: This is a placeholder - actual CUDA installation depends on the system
    echo "Please install CUDA toolkit manually: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
echo "✓ CUDA version: $CUDA_VERSION"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

echo "✓ Python3 found"

# Create virtual environment
VENV_DIR="/root/.openclaw/workspace/research/neural-rendering/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib tqdm
pip install opencv-python imageio

# Optional but useful
pip install tensorboard trimesh

echo ""
echo "=== Setup Complete ==="
echo "Virtual environment: $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "1. cd /root/.openclaw/workspace/research/neural-rendering"
echo "2. source venv/bin/activate"
echo "3. python src/gaussian.py  # Test basic implementation"

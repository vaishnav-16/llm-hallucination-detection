#!/bin/bash
set -e
echo "=== FastHalluCheck Setup ==="

# Create directories
mkdir -p data results figures report

# Install PyTorch with CUDA support first (RTX 4070 Mobile = CUDA 12.x)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"

# Install remaining dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

echo "=== Setup complete ==="

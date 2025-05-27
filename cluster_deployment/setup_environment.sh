#!/bin/bash

# Environment setup script for cluster deployment
# This script recreates the exact environment needed for your depth estimation training

set -e  # Exit on any error

echo "🔧 Setting up depth-pro environment on cluster..."

# Load required modules (adjust for your cluster)
echo "Loading cluster modules..."
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install miniconda/anaconda first."
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "depth-pro"; then
    echo "Removing existing depth-pro environment..."
    conda env remove -n depth-pro -y
fi

# Create environment from file (if available)
if [ -f "environment.yml" ]; then
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment.yml
else
    # Fallback: create environment manually
    echo "Creating conda environment manually..."
    conda create -n depth-pro python=3.9 -y
    conda activate depth-pro
    
    # Install PyTorch with CUDA support
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    # Install other dependencies
    pip install -r requirements.txt
fi

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate depth-pro

# Install ml-depth-pro package
echo "Installing ml-depth-pro package..."
if [ -d "ml-depth-pro" ]; then
    cd ml-depth-pro
    pip install -e .
    cd ..
else
    echo "❌ ml-depth-pro directory not found!"
    echo "Make sure to initialize the submodule: git submodule update --init --recursive"
    exit 1
fi

# Verify installation
echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')

import sys
sys.path.append('ml-depth-pro/src')
import depth_pro
print('✅ depth_pro imported successfully')

from monocular_depth.models.apple.depth import DepthProInference
print('✅ DepthProInference imported successfully')
"

echo "✅ Environment setup complete!"
echo "To activate: conda activate depth-pro" 
#!/bin/bash

# Environment setup script for cluster deployment
# This script recreates the exact environment needed for your depth estimation training

set -e  # Exit on any error

echo "🔧 Setting up depth-pro environment on cluster..."


# Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n depth-pro python=3.9 -y
conda activate depth-pro

conda install pip -y
pip install -r requirements.txt --upgrade
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate depth-pro

# Ensure git submodules are initialized
echo "Initializing submodules..."
git submodule update --init --recursive

# Check and install ml-depth-pro
if [ -d "ml-depth-pro" ]; then
    echo "Installing ml-depth-pro package..."
    pushd ml-depth-pro
    pip install -e .
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install ml-depth-pro"
        exit 1
    fi
    popd
else
    echo "❌ ml-depth-pro directory not found!"
    exit 1
fi
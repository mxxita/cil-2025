#!/bin/bash
#SBATCH --job-name=test_depth_pro
#SBATCH --output=logs/test_depth_pro_%j.out
#SBATCH --error=logs/test_depth_pro_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mem-per-gpu=16G
#SBATCH --partition=gpu

# Load required modules
module load gcc/8.2.0
module load python/3.8.5
module load cuda/11.7.0
module load cudnn/8.4.1.50

# Create logs directory
mkdir -p logs

# Print some useful information
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Python path: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}"
export CUDA_LAUNCH_BLOCKING=1

# Run the test script
python -m monocular_depth.tests.test_depth_pro

# Print job completion time
echo "Job finished at $(date)" 
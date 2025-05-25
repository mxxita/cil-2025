#!/bin/bash
#SBATCH --job-name=depth_estimation
#SBATCH --output=logs/depth_estimation_%j.out
#SBATCH --error=logs/depth_estimation_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@ethz.ch

# Load required modules
module load gcc/8.2.0
module load python/3.8.5
module load cuda/11.7.0
module load cudnn/8.4.1.50

# Create logs directory
mkdir -p logs

# Activate virtual environment (if you have one)
# source /path/to/your/venv/bin/activate

# Print some useful information
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Python path: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}"
export CUDA_LAUNCH_BLOCKING=1

# Create output directories
mkdir -p output/results
mkdir -p output/predictions
mkdir -p output/checkpoints

# Run the training script
python -m monocular_depth.main \
    --config config/config.yaml \
    --device cuda

# Print job completion time
echo "Job finished at $(date)" 
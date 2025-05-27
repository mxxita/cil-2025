#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=test_depth_estimation
#SBATCH --output=logs/test_depth_estimation_%j.out
#SBATCH --error=logs/test_depth_estimation_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mariberger@ethz.ch

# Create logs directory
mkdir -p logs

# Load stack modules for CUDA/GPU compatibility
module purge
module load stack/2024-05
module load gcc/13.2.0
module load cuda/12.2.1

# Clear Python-related environment variables to avoid conflicts
unset PYTHONPATH
unset PYTHONHOME

# Activate conda environment (conda's Python should take precedence)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate depth-pro

# Ensure conda's Python is used (override module Python)
export PATH="$CONDA_PREFIX/bin:$PATH"

# Print some useful information
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Verify depth_pro can be imported
echo "Testing depth_pro import..."
python -c "import depth_pro; print('✅ depth_pro imported successfully')" || echo "❌ depth_pro import failed"

nvidia-smi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}"
export CUDA_LAUNCH_BLOCKING=1
export DATA_DIR="/cluster/scratch/mariberger/monocular_depth/data"
export OUTPUT_DIR="${SLURM_SUBMIT_DIR}/outputs"

# Create output directories
mkdir -p outputs/results
mkdir -p outputs/predictions
mkdir -p outputs/checkpoints

# Run the training script
python -m monocular_depth.models.apple.main_test

# Print job completion time
echo "Job finished at $(date)" 
#!/bin/bash

#SBATCH --job-name=depth_estimation
#SBATCH --output=logs/depth_est_%j.out
#SBATCH --error=logs/depth_est_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --gres=gpu:1

# Optional: Specify GPU type if needed
# #SBATCH --constraint="gpu_mem:24GB"

unset PYTHONPATH
unset PYTHONHOME
# Create logs directory
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"

# Load required modules (adjust for your cluster)
module load stack/2024-05
module load gcc/13.2.0
module load python/3.9.18

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate depth-pro

# Set environment variables
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Print environment info
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs:"
nvidia-smi

# Create output directories
mkdir -p outputs/results
mkdir -p outputs/predictions
mkdir -p outputs/checkpoints

# Run training with cluster-optimized settings
python cluster_deployment/train_cluster.py \
    --config cluster_deployment/config_cluster.yaml \
    --job-id $SLURM_JOB_ID \
    --output-dir outputs \
    --use-gpu \
    --save-every 10 \
    --eval-every 5

echo "Job completed at: $(date)" 
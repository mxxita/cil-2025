#!/bin/bash

# Cluster deployment submission script
# Usage: ./submit_job.sh [config_file] [data_dir]

set -e  # Exit on any error

# Default values
CONFIG_FILE="${1:-cluster_deployment/config_cluster.yaml}"
DATA_DIR="${2:-/cluster/work/cvl/marber/data}"
JOB_NAME="depth_estimation_$(date +%Y%m%d_%H%M%S)"

echo "=== Cluster Deployment for Depth Estimation ==="
echo "Config file: $CONFIG_FILE"
echo "Data directory: $DATA_DIR"
echo "Job name: $JOB_NAME"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create necessary directories
mkdir -p logs outputs/results outputs/predictions outputs/checkpoints

# Update data paths in config (create temporary config)
TEMP_CONFIG="cluster_deployment/config_temp.yaml"
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# Replace data paths in temporary config
sed -i "s|train_dir:.*|train_dir: \"$DATA_DIR/train/\"|g" "$TEMP_CONFIG"
sed -i "s|test_dir:.*|test_dir: \"$DATA_DIR/test/\"|g" "$TEMP_CONFIG"
sed -i "s|train_list:.*|train_list: \"$DATA_DIR/train_list.txt\"|g" "$TEMP_CONFIG"
sed -i "s|test_list:.*|test_list: \"$DATA_DIR/test_list.txt\"|g" "$TEMP_CONFIG"

echo "Updated data paths in temporary config"

# Submit the job
echo "Submitting job to SLURM..."
JOB_ID=$(sbatch --job-name="$JOB_NAME" --parsable cluster_deployment/slurm_job.sh)

if [ $? -eq 0 ]; then
    echo "Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo "Job name: $JOB_NAME"
    echo ""
    echo "Monitor job status with:"
    echo "  squeue -j $JOB_ID"
    echo "  scontrol show job $JOB_ID"
    echo ""
    echo "View logs:"
    echo "  tail -f logs/depth_est_${JOB_ID}.out"
    echo "  tail -f logs/depth_est_${JOB_ID}.err"
    echo ""
    echo "Cancel job if needed:"
    echo "  scancel $JOB_ID"
else
    echo "Error: Failed to submit job"
    exit 1
fi

# Clean up temporary config
rm -f "$TEMP_CONFIG"

echo "Done!" 
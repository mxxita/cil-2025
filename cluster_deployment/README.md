# Cluster Deployment for Monocular Depth Estimation

This directory contains all the necessary files for deploying your depth estimation model on a computing cluster using SLURM.

## 🚀 Quick Start

1. **Setup your data on the cluster:**
   ```bash
   # Copy your data to cluster
   scp -r monocular_depth/data/ user@cluster:/cluster/work/cvl/marber/data/
   ```

2. **Make scripts executable:**
   ```bash
   chmod +x cluster_deployment/submit_job.sh
   chmod +x cluster_deployment/slurm_job.sh
   ```

3. **Submit your job:**
   ```bash
   ./cluster_deployment/submit_job.sh
   ```

## 📁 File Structure

```
cluster_deployment/
├── README.md              # This file
├── slurm_job.sh           # SLURM batch script
├── config_cluster.yaml    # Training configuration
├── train_cluster.py       # Enhanced training script
├── metrics.py             # Evaluation metrics
├── checkpointing.py       # Checkpoint management
└── submit_job.sh          # Job submission script
```

## ⚙️ Configuration

### SLURM Job Settings (`slurm_job.sh`)

Key parameters you might need to modify:

```bash
#SBATCH --time=24:00:00       # Maximum runtime
#SBATCH --partition=gpu       # GPU partition name
#SBATCH --gres=gpu:1          # Number of GPUs
#SBATCH --mem=32G             # Memory allocation
#SBATCH --cpus-per-task=8     # CPU cores
```

### Training Configuration (`config_cluster.yaml`)

Main settings to adjust:

```yaml
training:
  num_epochs: 50              # Number of training epochs
  batch_size: 8               # Batch size (adjust for GPU memory)
  learning_rate: 1e-4         # Learning rate

data:
  num_workers: 8              # Data loading workers
  train_split: 0.85           # Train/validation split

hardware:
  mixed_precision: true       # Use mixed precision training
  device: "auto"              # auto, cpu, or cuda
```

## 📊 Evaluation Metrics

The system computes standard depth estimation metrics:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error  
- **siRMSE**: Scale-Invariant RMSE
- **LogRMSE**: Log Root Mean Square Error
- **REL**: Mean Relative Error
- **Delta1/2/3**: Threshold accuracy (δ < 1.25^n)

## 🔄 Usage Examples

### Basic Usage
```bash
# Submit with default settings
./cluster_deployment/submit_job.sh
```

### Custom Configuration
```bash
# Use custom config file
./cluster_deployment/submit_job.sh my_config.yaml

# Specify data directory
./cluster_deployment/submit_job.sh config_cluster.yaml /path/to/my/data
```

### Monitor Training
```bash
# Check job status
squeue -u $USER

# View training logs
tail -f logs/depth_est_JOBID.out

# View error logs  
tail -f logs/depth_est_JOBID.err

# Monitor GPU usage
ssh node_name nvidia-smi
```

## 📈 Output Structure

After training, you'll find:

```
outputs/
├── results/                # Training results and metrics
├── predictions/            # Test set predictions  
├── checkpoints/            # Model checkpoints
│   ├── best.pth           # Best model
│   ├── final.pth          # Final model
│   └── checkpoint_epoch_*.pth
└── logs/                   # Training logs
    ├── tensorboard/        # TensorBoard logs
    └── train_JOBID.log     # Training log file
```

## 🛠️ Advanced Usage

### Resume Training
```python
# Load from checkpoint
python cluster_deployment/train_cluster.py \
    --config config_cluster.yaml \
    --resume outputs/checkpoints/checkpoint_epoch_20.pth
```

### Hyperparameter Sweeps
```bash
# Create multiple configs with different hyperparameters
for lr in 1e-4 1e-5 1e-3; do
    sed "s/learning_rate: 1e-4/learning_rate: $lr/" config_cluster.yaml > config_lr_$lr.yaml
    ./submit_job.sh config_lr_$lr.yaml
done
```

### Multi-GPU Training
```bash
# Modify slurm_job.sh for multi-GPU
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2

# Use PyTorch DDP in train_cluster.py
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `batch_size` in config
   - Reduce `num_workers`
   - Set `mixed_precision: true`

2. **Job Timeout**
   - Increase `#SBATCH --time`
   - Reduce `num_epochs`
   - Save checkpoints more frequently

3. **Module Not Found**
   - Check conda environment activation
   - Verify PYTHONPATH settings
   - Ensure all dependencies are installed

4. **Data Loading Errors**
   - Verify data paths in config
   - Check file permissions
   - Ensure data format matches expected structure

### Debug Mode
```bash
# Run interactively for debugging
srun --pty --gres=gpu:1 --mem=16G --time=1:00:00 bash
conda activate depth-pro
python cluster_deployment/train_cluster.py --config config_cluster.yaml
```

## 📋 Cluster-Specific Setup

### ETH Euler Cluster
```bash
# Load modules
module load gcc/8.2.0 python_gpu/3.9.9 cuda/11.2.2

# Partition
#SBATCH --partition=gpu.24h
```

### Other Clusters
- Modify module loading in `slurm_job.sh`
- Adjust partition names
- Update data path conventions

## 🎯 Performance Optimization

### GPU Utilization
- Monitor with `nvidia-smi`
- Adjust batch size to maximize GPU memory usage
- Use mixed precision training
- Consider gradient accumulation for larger effective batch sizes

### Data Loading
- Increase `num_workers` (typically 4-8)
- Use `pin_memory: true` for GPU training
- Consider data caching for repeated experiments

### Training Speed
- Use compiled models (`compile_model: true`) for PyTorch 2.0+
- Enable mixed precision training
- Use appropriate learning rate scheduling

## 📞 Support

For cluster-specific issues:
1. Check cluster documentation
2. Contact cluster support team
3. Review SLURM documentation

For code issues:
1. Check the training logs in `outputs/logs/`
2. Verify configuration settings
3. Test with smaller datasets first

## 🔄 Version Updates

To update the deployment:
1. Pull latest code changes
2. Update conda environment if needed
3. Resubmit jobs with new code

---

**Happy Training! 🚀** 
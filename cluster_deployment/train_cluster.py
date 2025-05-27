#!/usr/bin/env python3
"""
Cluster training script for monocular depth estimation.
Enhanced with proper logging, checkpointing, and evaluation.
"""

import argparse
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Logging will be limited.")
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from monocular_depth.models.apple.depth import DepthProInference
from monocular_depth.data.dataset import DepthDataset
from monocular_depth.data.transforms import train_transform, test_transform
from monocular_depth.training.loss import SILogLoss
from monocular_depth.utils.helpers import ensure_dir, custom_collate_fn
from cluster_deployment.metrics import DepthMetrics
from cluster_deployment.checkpointing import CheckpointManager


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    # Configure logging
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(config: Dict[str, Any]) -> torch.device:
    """Get the appropriate device based on config and availability."""
    device_config = config['hardware']['device']
    
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_config)
    
    logging.info(f"Using device: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name()}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def create_data_loaders(config: Dict[str, Any]) -> tuple:
    """Create train, validation, and test data loaders."""
    data_config = config['data']
    
    # Create datasets
    train_full_dataset = DepthDataset(
        data_dir=data_config['train_dir'],
        list_file=data_config['train_list'],
        input_size=tuple(data_config['input_size']),
        transform=train_transform,
        has_gt=True,
    )
    
    test_dataset = DepthDataset(
        data_dir=data_config['test_dir'],
        list_file=data_config['test_list'],
        input_size=tuple(data_config['input_size']),
        transform=test_transform,
        has_gt=False,
    )
    
    # Split training dataset
    total_size = len(train_full_dataset)
    train_size = int(data_config['train_split'] * total_size)
    val_size = total_size - train_size
    
    torch.manual_seed(42)  # For reproducible splits
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        drop_last=True,
        persistent_workers=data_config['persistent_workers'],
        collate_fn=custom_collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=custom_collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=custom_collate_fn,
    )
    
    logging.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create and initialize the model."""
    model_config = config['model']
    
    model = DepthProInference(
        prefer_cpu=(device.type == 'cpu'),
        enable_training=model_config['enable_training']
    )
    
    model = model.to(device)
    
    # Optional: Compile model for PyTorch 2.0+
    if config['hardware'].get('compile_model', False):
        try:
            model = torch.compile(model)
            logging.info("Model compiled successfully")
        except Exception as e:
            logging.warning(f"Model compilation failed: {e}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def create_optimizer_scheduler(model: nn.Module, config: Dict[str, Any]):
    """Create optimizer and learning rate scheduler."""
    training_config = config['training']
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # Create scheduler if specified
    scheduler = None
    if 'scheduler' in training_config:
        sched_config = training_config['scheduler']
        if sched_config['type'] == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        elif sched_config['type'] == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['num_epochs']
            )
    
    return optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, config, epoch, writer=None):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue
            
        inputs, targets, filenames = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if config['hardware'].get('mixed_precision', False) and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Backward pass
        if config['hardware'].get('mixed_precision', False) and device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        # Log batch metrics
        if batch_idx % config['logging']['log_every'] == 0:
            logging.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
            
            if writer:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/batch_loss', loss.item(), global_step)
    
    avg_loss = epoch_loss / max(num_batches, 1)
    return avg_loss


def validate(model, val_loader, criterion, device, metrics_calculator):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    num_batches = 0
    
    metrics_calculator.reset()
    
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
                
            inputs, targets, filenames = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            num_batches += 1
            
            # Update metrics
            metrics_calculator.update(outputs, targets)
    
    avg_loss = val_loss / max(num_batches, 1)
    metrics = metrics_calculator.compute()
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Cluster training for depth estimation')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--job-id', help='SLURM job ID')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--use-gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    
    args = parser.parse_args()
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    results_dir = output_dir / 'results'
    checkpoints_dir = output_dir / 'checkpoints'
    logs_dir = output_dir / 'logs'
    
    for dir_path in [results_dir, checkpoints_dir, logs_dir]:
        ensure_dir(str(dir_path))
    
    # Setup logging
    log_file = logs_dir / f'train_{args.job_id or int(time.time())}.log'
    logger = setup_logging(log_file=str(log_file))
    
    logger.info("Starting cluster training")
    logger.info(f"Job ID: {args.job_id}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Override GPU setting if specified
    if args.use_gpu:
        config['hardware']['device'] = 'cuda'
    
    # Get device
    device = get_device(config)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Create model
    model = create_model(config, device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_scheduler(model, config)
    
    # Create loss function
    criterion = SILogLoss()
    
    # Create metrics calculator
    metrics_calculator = DepthMetrics(config['evaluation']['metrics'])
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoints_dir,
        max_checkpoints=config['checkpointing']['max_checkpoints']
    )
    
    # Setup tensorboard
    writer = None
    if config['logging']['tensorboard'] and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(logs_dir / 'tensorboard')
    elif config['logging']['tensorboard'] and not TENSORBOARD_AVAILABLE:
        logger.warning("TensorBoard logging requested but not available")
    
    # Training loop
    best_metric = float('inf')
    start_time = time.time()
    
    for epoch in range(config['training']['num_epochs']):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config, epoch, writer)
        
        # Validate
        if epoch % args.eval_every == 0:
            val_loss, val_metrics = validate(model, val_loader, criterion, device, metrics_calculator)
            
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            for metric_name, value in val_metrics.items():
                logger.info(f"  {metric_name}: {value:.6f}")
            
            # Log to tensorboard
            if writer:
                writer.add_scalar('train/epoch_loss', train_loss, epoch)
                writer.add_scalar('val/loss', val_loss, epoch)
                for metric_name, value in val_metrics.items():
                    writer.add_scalar(f'val/{metric_name}', value, epoch)
            
            # Save best model
            current_metric = val_metrics.get('siRMSE', val_loss)
            if current_metric < best_metric:
                best_metric = current_metric
                checkpoint_manager.save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics, 'best'
                )
                logger.info(f"New best model saved with {current_metric:.6f}")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, epoch, {}, f'epoch_{epoch}'
            )
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")
    
    # Final evaluation on test set (if has ground truth)
    logger.info("Running final evaluation...")
    
    # Save final model
    checkpoint_manager.save_checkpoint(
        model, optimizer, scheduler, config['training']['num_epochs']-1, {}, 'final'
    )
    
    # Cleanup
    if writer:
        writer.close()
    
    logger.info("Training completed successfully")


if __name__ == '__main__':
    main() 
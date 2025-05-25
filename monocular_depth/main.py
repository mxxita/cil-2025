import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
from typing import Dict, Any

from .data.dataset import DepthDataset
from .data.transforms import get_transforms
from .models.unet import SimpleUNet
from .training.trainer import Trainer
from .utils.helpers import (
    ensure_dir, load_config, get_device,
    clear_gpu_memory
)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train depth estimation model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to train on (cuda or cpu)')
    return parser.parse_args()

def create_data_loaders(config: Dict[str, Any]) -> tuple:
    """
    Create data loaders for training, validation and testing.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get data paths from config
    data_config = config['data']
    train_dir = data_config['train_dir']
    test_dir = data_config['test_dir']
    train_list = data_config['train_list']
    test_list = data_config['test_list']
    
    # Get training parameters
    train_config = config['training']
    batch_size = train_config['batch_size']
    num_workers = train_config['num_workers']
    pin_memory = train_config['pin_memory']
    train_val_split = train_config['train_val_split']
    
    # Get model parameters
    model_config = config['model']
    input_size = tuple(model_config['input_size'])
    
    # Get augmentation parameters
    aug_config = config['augmentation']
    
    # Create transforms
    train_transform, train_depth_transform = get_transforms(
        input_size=input_size,
        is_train=True,
        color_jitter_params=aug_config['color_jitter'],
        normalize_params=aug_config['normalize']
    )
    
    test_transform, test_depth_transform = get_transforms(
        input_size=input_size,
        is_train=False,
        normalize_params=aug_config['normalize']
    )
    
    # Create datasets
    train_full_dataset = DepthDataset(
        data_dir=train_dir,
        list_file=train_list,
        transform=train_transform,
        target_transform=train_depth_transform,
        has_gt=True
    )
    
    test_dataset = DepthDataset(
        data_dir=test_dir,
        list_file=test_list,
        transform=test_transform,
        has_gt=False
    )
    
    # Split training dataset
    total_size = len(train_full_dataset)
    train_size = int(train_val_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        train_full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def main():
    """Main training script."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    ensure_dir(config['data']['output_dir'])
    ensure_dir(config['data']['results_dir'])
    ensure_dir(config['data']['predictions_dir'])
    
    # Get device
    device = get_device(args.device)
    
    # Clear GPU memory
    clear_gpu_memory()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Create model
    model = SimpleUNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels']
    )
    
    # Move model to device and wrap with DataParallel if using multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Create loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config,
        save_dir=config['data']['output_dir']
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train(config['training']['num_epochs'])
    
    # Generate test predictions
    print("Generating test predictions...")
    trainer.generate_predictions(
        test_loader=test_loader,
        output_dir=config['data']['predictions_dir']
    )
    
    print(f"Results saved to {config['data']['results_dir']}")
    print(f"Test predictions saved to {config['data']['predictions_dir']}")

if __name__ == '__main__':
    main() 
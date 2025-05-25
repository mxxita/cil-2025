import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm
import yaml
from ..utils.helpers import ensure_dir
from .metrics import DepthMetrics

class Trainer:
    """Trainer class for depth estimation model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        save_dir: str
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            config: Training configuration
            save_dir: Directory to save checkpoints and results
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.save_dir = save_dir
        
        # Create directories
        self.checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        self.results_dir = os.path.join(save_dir, 'results')
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.results_dir)
        
        # Initialize metrics
        self.metrics = DepthMetrics(device)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for inputs, targets, _ in tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch + 1}"):
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item() * inputs.size(0)
        
        # Calculate average loss
        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss
    
    def validate(self, save_samples: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Args:
            save_samples: Whether to save sample predictions
            
        Returns:
            Tuple of (average validation loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0.0
        self.metrics.reset()
        
        with torch.no_grad():
            for inputs, targets, filenames in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update statistics
                total_loss += loss.item() * inputs.size(0)
                
                # Update metrics
                self.metrics.update(
                    outputs, targets,
                    save_samples=save_samples,
                    save_dir=self.results_dir if save_samples else None,
                    rgb_input=inputs,
                    filenames=filenames,
                    max_samples=self.config['evaluation']['save_samples']
                )
        
        # Calculate average loss and metrics
        avg_loss = total_loss / len(self.val_loader.dataset)
        metrics = self.metrics.compute()
        
        return avg_loss, metrics
    
    def train(self, num_epochs: int) -> Dict[str, Any]:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(save_samples=True)
            history['val_loss'].append(val_loss)
            history['metrics'].append(metrics)
            
            # Print progress
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print("Validation Metrics:")
            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")
            
            # Save checkpoint if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth')
                print(f"New best model saved at epoch {epoch + 1}")
            
            # Save regular checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Save metrics to file
            self.save_metrics(metrics, epoch)
        
        # Load best model for final evaluation
        self.load_checkpoint('best_model.pth')
        print(f"\nBest model was from epoch {self.best_epoch + 1}")
        
        return history
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Name of the checkpoint file
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config
        }
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
    
    def load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            filename: Name of the checkpoint file
        """
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, filename))
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
    
    def save_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Save metrics to file.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Current epoch number
        """
        metrics_file = os.path.join(self.results_dir, 'metrics.yaml')
        
        # Load existing metrics if file exists
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                all_metrics = yaml.safe_load(f) or {}
        else:
            all_metrics = {}
        
        # Update metrics for current epoch
        all_metrics[f'epoch_{epoch + 1}'] = metrics
        
        # Save updated metrics
        with open(metrics_file, 'w') as f:
            yaml.dump(all_metrics, f, default_flow_style=False)
    
    def generate_predictions(self, test_loader: DataLoader, output_dir: str) -> None:
        """
        Generate predictions for test set.
        
        Args:
            test_loader: Test data loader
            output_dir: Directory to save predictions
        """
        self.model.eval()
        ensure_dir(output_dir)
        
        with torch.no_grad():
            for inputs, filenames in tqdm(test_loader, desc="Generating Test Predictions"):
                # Move data to device
                inputs = inputs.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Save predictions
                for i, filename in enumerate(filenames):
                    # Get filename without extension
                    filename = filename.split(' ')[1]
                    
                    # Save depth map prediction
                    depth_pred = outputs[i].cpu().squeeze().numpy()
                    np.save(os.path.join(output_dir, f"{filename}"), depth_pred) 
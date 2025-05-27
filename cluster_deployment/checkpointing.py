"""
Checkpoint management for training.
Handles saving and loading model states, optimizer states, and training metadata.
"""

import os
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import glob


class CheckpointManager:
    """Manage model checkpoints during training."""
    
    def __init__(self, checkpoint_dir: Union[str, Path], max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        metrics: Dict[str, float],
        checkpoint_name: str = None
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler state to save
            epoch: Current epoch number
            metrics: Dictionary of metrics for this checkpoint
            checkpoint_name: Custom name for checkpoint, if None uses epoch number
            
        Returns:
            Path to saved checkpoint
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{epoch}"
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pth"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'model_class': model.__class__.__name__,
        }
        
        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint_data['scheduler_class'] = scheduler.__class__.__name__
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Clean up old checkpoints if needed
            if checkpoint_name.startswith("checkpoint_epoch_"):
                self._cleanup_old_checkpoints()
                
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load tensors on
            
        Returns:
            Dictionary with loaded metadata (epoch, metrics, etc.)
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint
            if device is None:
                checkpoint_data = torch.load(checkpoint_path)
            else:
                checkpoint_data = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint_data['model_state_dict'])
            self.logger.info(f"Model state loaded from {checkpoint_path}")
            
            # Load optimizer state if provided
            if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                self.logger.info("Optimizer state loaded")
            
            # Load scheduler state if provided
            if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                self.logger.info("Scheduler state loaded")
            
            # Return metadata
            metadata = {
                'epoch': checkpoint_data.get('epoch', 0),
                'metrics': checkpoint_data.get('metrics', {}),
                'model_class': checkpoint_data.get('model_class', ''),
            }
            
            self.logger.info(f"Checkpoint loaded successfully from epoch {metadata['epoch']}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Find the latest checkpoint file.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints found
        """
        checkpoint_pattern = str(self.checkpoint_dir / "checkpoint_epoch_*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            return None
        
        # Extract epoch numbers and find the latest
        latest_epoch = -1
        latest_file = None
        
        for file_path in checkpoint_files:
            try:
                # Extract epoch number from filename
                filename = Path(file_path).stem
                epoch_str = filename.split('_')[-1]
                epoch = int(epoch_str)
                
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_file = file_path
                    
            except (ValueError, IndexError):
                continue
        
        if latest_file:
            self.logger.info(f"Latest checkpoint found: {latest_file} (epoch {latest_epoch})")
        
        return latest_file
    
    def find_best_checkpoint(self) -> Optional[str]:
        """
        Find the best checkpoint file.
        
        Returns:
            Path to best checkpoint or None if not found
        """
        best_checkpoint = self.checkpoint_dir / "best.pth"
        
        if best_checkpoint.exists():
            return str(best_checkpoint)
        
        return None
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints with their metadata.
        
        Returns:
            List of dictionaries with checkpoint information
        """
        checkpoint_pattern = str(self.checkpoint_dir / "*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        checkpoints = []
        
        for file_path in checkpoint_files:
            try:
                # Load just the metadata
                checkpoint_data = torch.load(file_path, map_location='cpu')
                
                checkpoint_info = {
                    'path': file_path,
                    'filename': Path(file_path).name,
                    'epoch': checkpoint_data.get('epoch', 0),
                    'metrics': checkpoint_data.get('metrics', {}),
                    'model_class': checkpoint_data.get('model_class', ''),
                    'size_mb': Path(file_path).stat().st_size / (1024 * 1024)
                }
                
                checkpoints.append(checkpoint_info)
                
            except Exception as e:
                self.logger.warning(f"Could not read checkpoint {file_path}: {e}")
                continue
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x['epoch'])
        return checkpoints
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoint files, keeping only the most recent ones."""
        checkpoint_pattern = str(self.checkpoint_dir / "checkpoint_epoch_*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x))
        
        # Remove oldest files
        files_to_remove = checkpoint_files[:-self.max_checkpoints]
        
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                self.logger.info(f"Removed old checkpoint: {Path(file_path).name}")
            except Exception as e:
                self.logger.warning(f"Could not remove checkpoint {file_path}: {e}")
    
    def save_config(self, config: Dict[str, Any]):
        """Save training configuration."""
        config_path = self.checkpoint_dir / "config.yaml"
        
        try:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            self.logger.info(f"Config saved: {config_path}")
        except Exception as e:
            self.logger.warning(f"Could not save config: {e}")


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics where lower is better, 'max' for metrics where higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda score, best: score < (best - min_delta)
        else:
            self.is_better = lambda score, best: score > (best + min_delta)
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            score: Current metric score
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop 
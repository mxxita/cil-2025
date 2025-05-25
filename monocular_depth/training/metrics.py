import torch
import numpy as np
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import os

class DepthMetrics:
    """Metrics for evaluating depth estimation models."""
    
    def __init__(self, device: torch.device):
        """
        Initialize metrics calculator.
        
        Args:
            device: Device to perform calculations on
        """
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all metrics to zero."""
        self.mae = 0.0
        self.rmse = 0.0
        self.rel = 0.0
        self.delta1 = 0.0
        self.delta2 = 0.0
        self.delta3 = 0.0
        self.sirmse = 0.0
        self.total_samples = 0
        self.total_pixels = 0
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        save_samples: bool = False,
        save_dir: Optional[str] = None,
        rgb_input: Optional[torch.Tensor] = None,
        filenames: Optional[list] = None,
        max_samples: int = 5
    ) -> None:
        """
        Update metrics with a batch of predictions.
        
        Args:
            pred: Predicted depth maps
            target: Ground truth depth maps
            save_samples: Whether to save sample visualizations
            save_dir: Directory to save sample visualizations
            rgb_input: RGB input images for visualization
            filenames: List of filenames for saving samples
            max_samples: Maximum number of samples to save
        """
        batch_size = pred.size(0)
        self.total_samples += batch_size
        
        if self.total_pixels == 0:
            self.total_pixels = target.numel() // batch_size
        
        # Move tensors to device if needed
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        # Calculate absolute difference
        abs_diff = torch.abs(pred - target)
        
        # Update metrics
        self.mae += torch.sum(abs_diff).item()
        self.rmse += torch.sum(torch.pow(abs_diff, 2)).item()
        self.rel += torch.sum(abs_diff / (target + 1e-6)).item()
        
        # Calculate thresholded accuracy
        max_ratio = torch.max(pred / (target + 1e-6), target / (pred + 1e-6))
        self.delta1 += torch.sum(max_ratio < 1.25).item()
        self.delta2 += torch.sum(max_ratio < 1.25**2).item()
        self.delta3 += torch.sum(max_ratio < 1.25**3).item()
        
        # Calculate scale-invariant RMSE
        for i in range(batch_size):
            pred_np = pred[i].cpu().squeeze().numpy()
            target_np = target[i].cpu().squeeze().numpy()
            
            # Handle invalid values
            valid_mask = target_np > 1e-6
            if not np.any(valid_mask):
                continue
            
            # Calculate scale-invariant error
            log_pred = np.log(np.maximum(pred_np[valid_mask], 1e-6))
            log_target = np.log(target_np[valid_mask])
            diff = log_pred - log_target
            diff_mean = np.mean(diff)
            
            self.sirmse += np.sqrt(np.mean((diff - diff_mean) ** 2))
        
        # Save sample visualizations
        if save_samples and save_dir and rgb_input is not None and filenames is not None:
            self._save_samples(
                rgb_input, pred, target, filenames,
                save_dir, max_samples
            )
    
    def _save_samples(
        self,
        rgb: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        filenames: list,
        save_dir: str,
        max_samples: int
    ) -> None:
        """
        Save sample visualizations.
        
        Args:
            rgb: RGB input images
            pred: Predicted depth maps
            target: Ground truth depth maps
            filenames: List of filenames
            save_dir: Directory to save visualizations
            max_samples: Maximum number of samples to save
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(min(len(filenames), max_samples)):
            # Convert tensors to numpy arrays
            rgb_np = rgb[i].cpu().permute(1, 2, 0).numpy()
            pred_np = pred[i].cpu().squeeze().numpy()
            target_np = target[i].cpu().squeeze().numpy()
            
            # Normalize RGB for visualization
            rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-6)
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(rgb_np)
            plt.title("RGB Input")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(target_np, cmap='plasma')
            plt.title("Ground Truth Depth")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(pred_np, cmap='plasma')
            plt.title("Predicted Depth")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{filenames[i]}.png"))
            plt.close()
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        total_pixels = self.total_samples * self.total_pixels
        
        metrics = {
            'MAE': self.mae / total_pixels,
            'RMSE': np.sqrt(self.rmse / total_pixels),
            'REL': self.rel / total_pixels,
            'Delta1': self.delta1 / total_pixels,
            'Delta2': self.delta2 / total_pixels,
            'Delta3': self.delta3 / total_pixels,
            'siRMSE': self.sirmse / self.total_samples
        }
        
        return metrics 
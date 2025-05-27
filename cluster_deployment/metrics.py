"""
Comprehensive depth estimation metrics.
Standard metrics used in depth estimation literature.
"""

import torch
import numpy as np
from typing import List, Dict, Optional


class DepthMetrics:
    """Calculate standard depth estimation metrics."""
    
    def __init__(self, metric_names: List[str]):
        """
        Initialize metrics calculator.
        
        Args:
            metric_names: List of metric names to compute
                         ['MAE', 'RMSE', 'siRMSE', 'LogRMSE', 'REL', 'Delta1', 'Delta2', 'Delta3']
        """
        self.metric_names = metric_names
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.count = 0
        self.sum_mae = 0.0
        self.sum_mse = 0.0
        self.sum_rel = 0.0
        self.sum_delta1 = 0.0
        self.sum_delta2 = 0.0
        self.sum_delta3 = 0.0
        self.sum_log_mse = 0.0
        self.sum_si_mse = 0.0
        self.sum_si_mean_sq = 0.0
        self.total_valid_pixels = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a batch of predictions and targets.
        Online computation to avoid memory accumulation.
        
        Args:
            pred: Predicted depth maps (B, 1, H, W) or (B, H, W)
            target: Target depth maps (B, 1, H, W) or (B, H, W)
        """
        # Ensure tensors are on CPU and have same shape
        pred = pred.detach().cpu()
        target = target.detach().cpu()
        
        # Remove channel dimension if present
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        
        # Get valid mask
        mask = self._get_valid_mask(pred, target)
        if not mask.any():
            return
            
        pred_valid = pred[mask]
        target_valid = target[mask]
        n_valid = mask.sum().item()
        
        # Update counters
        self.count += pred.shape[0]  # Number of images
        self.total_valid_pixels += n_valid
        
        # Compute and accumulate metrics online
        if 'MAE' in self.metric_names:
            mae_batch = torch.mean(torch.abs(pred_valid - target_valid))
            self.sum_mae += mae_batch.item() * n_valid
            
        if 'RMSE' in self.metric_names:
            mse_batch = torch.mean((pred_valid - target_valid) ** 2)
            self.sum_mse += mse_batch.item() * n_valid
            
        if 'REL' in self.metric_names:
            rel_batch = torch.mean(torch.abs(pred_valid - target_valid) / target_valid)
            self.sum_rel += rel_batch.item() * n_valid
            
        if any(name in self.metric_names for name in ['Delta1', 'Delta2', 'Delta3']):
            ratio = torch.max(pred_valid / target_valid, target_valid / pred_valid)
            if 'Delta1' in self.metric_names:
                delta1_batch = torch.mean((ratio < 1.25).float())
                self.sum_delta1 += delta1_batch.item() * n_valid
            if 'Delta2' in self.metric_names:
                delta2_batch = torch.mean((ratio < 1.25**2).float())
                self.sum_delta2 += delta2_batch.item() * n_valid
            if 'Delta3' in self.metric_names:
                delta3_batch = torch.mean((ratio < 1.25**3).float())
                self.sum_delta3 += delta3_batch.item() * n_valid
                
        if 'LogRMSE' in self.metric_names:
            log_pred = torch.log(pred_valid)
            log_target = torch.log(target_valid)
            log_mse_batch = torch.mean((log_pred - log_target) ** 2)
            self.sum_log_mse += log_mse_batch.item() * n_valid
            
        if 'siRMSE' in self.metric_names:
            log_pred = torch.log(pred_valid)
            log_target = torch.log(target_valid)
            diff = log_pred - log_target
            diff_mean = torch.mean(diff)
            si_error = diff - diff_mean
            si_mse_batch = torch.mean(si_error ** 2)
            self.sum_si_mse += si_mse_batch.item() * n_valid
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all requested metrics from accumulated values.
        
        Returns:
            Dictionary of metric names and values
        """
        if self.total_valid_pixels == 0:
            return {name: 0.0 for name in self.metric_names}
        
        metrics = {}
        
        for metric_name in self.metric_names:
            if metric_name == 'MAE':
                metrics[metric_name] = self.sum_mae / self.total_valid_pixels
            elif metric_name == 'RMSE':
                metrics[metric_name] = np.sqrt(self.sum_mse / self.total_valid_pixels)
            elif metric_name == 'siRMSE':
                metrics[metric_name] = np.sqrt(self.sum_si_mse / self.total_valid_pixels)
            elif metric_name == 'LogRMSE':
                metrics[metric_name] = np.sqrt(self.sum_log_mse / self.total_valid_pixels)
            elif metric_name == 'REL':
                metrics[metric_name] = self.sum_rel / self.total_valid_pixels
            elif metric_name == 'Delta1':
                metrics[metric_name] = self.sum_delta1 / self.total_valid_pixels
            elif metric_name == 'Delta2':
                metrics[metric_name] = self.sum_delta2 / self.total_valid_pixels
            elif metric_name == 'Delta3':
                metrics[metric_name] = self.sum_delta3 / self.total_valid_pixels
            else:
                metrics[metric_name] = 0.0
        
        return metrics
    
    def _get_valid_mask(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Get mask for valid depth values."""
        valid_target = target > eps
        valid_pred = pred > eps
        finite_pred = torch.isfinite(pred)
        finite_target = torch.isfinite(target)
        return valid_target & valid_pred & finite_pred & finite_target


class OnlineMetrics:
    """Online computation of metrics for streaming evaluation."""
    
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.reset()
    
    def reset(self):
        """Reset all accumulated values."""
        self.count = 0
        self.sum_mae = 0.0
        self.sum_mse = 0.0
        self.sum_rel = 0.0
        self.sum_delta1 = 0.0
        self.sum_delta2 = 0.0
        self.sum_delta3 = 0.0
        self.sum_log_error = 0.0
        self.sum_si_error = 0.0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with new batch."""
        # Implementation for online metric computation
        # This is more memory efficient for large datasets
        pass
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        if self.count == 0:
            return {name: 0.0 for name in self.metric_names}
        
        metrics = {}
        if 'MAE' in self.metric_names:
            metrics['MAE'] = self.sum_mae / self.count
        if 'RMSE' in self.metric_names:
            metrics['RMSE'] = np.sqrt(self.sum_mse / self.count)
        # Add other metrics as needed
        
        return metrics 
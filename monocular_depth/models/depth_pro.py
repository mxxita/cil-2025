import torch
import torch.nn as nn
from depth_pro import create_model_and_transforms
from typing import Tuple

class ModelOutput:
    """Simple container to match expected output format"""
    def __init__(self, predicted_depth):
        self.predicted_depth = predicted_depth

class DepthProInference(nn.Module):
    """Wrapper class for DepthPro model inference, matching UNet interface."""
    
    def __init__(self):
        """Initialize the DepthPro model."""
        super().__init__()
        self.model, self.transform = create_model_and_transforms()
        self.model.eval()
    
    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            ModelOutput object with predicted_depth attribute
        """
        # Perform inference without f_px parameter (let DepthPro auto-detect)
        with torch.no_grad():
            prediction = self.model.infer(x)
            depth_map = prediction["depth"]  # Depth in meters
            
        # Ensure proper batch dimensions
        if depth_map.dim() == 2:
            # Single image: (H, W) -> (1, 1, H, W)
            depth_map = depth_map.unsqueeze(0).unsqueeze(0)
        elif depth_map.dim() == 3:
            # Batch of images: (B, H, W) -> (B, 1, H, W)
            depth_map = depth_map.unsqueeze(1)
        
        return ModelOutput(depth_map) 
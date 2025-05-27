import torch
import torch.nn as nn
from depth_pro import create_model_and_transforms
from typing import Tuple

class DepthProInference(nn.Module):
    """Wrapper class for DepthPro model inference, matching UNet interface."""
    
    def __init__(self):
        """Initialize the DepthPro model."""
        super().__init__()
        self.model, self.transform = create_model_and_transforms()
        self.model.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Perform inference
        with torch.no_grad():
            prediction = self.model.infer(x, f_px=1000.0)
            depth_map = prediction["depth"]  # Depth in meters
            
        # Scale depth map to [0, 10] range to match UNet output
        depth = 1.0 / (depth_map + 1e-6)
        depth = torch.clamp(depth, 0.0, 10.0)  # Optional: match ground truth range
        
        return depth_map 
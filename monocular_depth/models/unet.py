import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .blocks import UNetBlock

class SimpleUNet(nn.Module):
    """Simple U-Net architecture for monocular depth estimation."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        use_batch_norm: bool = True,
        dropout_rate: Optional[float] = None
    ):
        """
        Initialize Simple U-Net.
        
        Args:
            in_channels: Number of input channels (RGB)
            out_channels: Number of output channels (depth)
            base_channels: Number of base channels in the first layer
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (None for no dropout)
        """
        super().__init__()
        
        # Encoder blocks
        self.enc1 = UNetBlock(in_channels, base_channels, use_batch_norm, dropout_rate)
        self.enc2 = UNetBlock(base_channels, base_channels * 2, use_batch_norm, dropout_rate)
        
        # Decoder blocks with skip connections
        self.dec2 = UNetBlock(base_channels * 2 + base_channels, base_channels, use_batch_norm, dropout_rate)
        self.dec1 = UNetBlock(base_channels, base_channels // 2, use_batch_norm, dropout_rate)
        
        # Final layer
        self.final = nn.Conv2d(base_channels // 2, out_channels, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Encoder
        enc1 = self.enc1(x)  # Save for skip connection
        x = self.pool(enc1)
        
        x = self.enc2(x)
        
        # Decoder with skip connections
        x = F.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc1], dim=1)  # Skip connection
        x = self.dec2(x)
        
        x = self.dec1(x)
        x = self.final(x)
        
        # Output non-negative depth values using sigmoid
        x = torch.sigmoid(x) * 10  # Scale to reasonable depth range
        
        return x 
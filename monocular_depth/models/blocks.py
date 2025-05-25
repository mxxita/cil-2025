import torch
import torch.nn as nn
from typing import Optional

class UNetBlock(nn.Module):
    """Basic block for U-Net architecture."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        dropout_rate: Optional[float] = None
    ):
        """
        Initialize U-Net block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (None for no dropout)
        """
        super().__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        
        # Dropout layer
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate is not None else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # First convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second convolution block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x 
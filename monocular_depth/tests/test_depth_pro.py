"""Test script to debug DepthPro model issues."""

import torch
import numpy as np
from PIL import Image

from monocular_depth.models.apple.depth_pro import DepthProInference


def test_depth_pro_basic():
    """Test basic DepthPro functionality."""
    print("Testing DepthPro model...")
    
    # Create model
    model = DepthProInference()
    print(f"Model device: {model.device}")
    
    # Create test input
    batch_size = 1
    channels = 3
    height = 224
    width = 224
    
    # Create random input tensor
    x = torch.randn(batch_size, channels, height, width)
    print(f"Input shape: {x.shape}, dtype: {x.dtype}, contiguous: {x.is_contiguous()}")
    
    try:
        # Test forward pass
        with torch.no_grad():
            output = model(x)
        print(f"Output shape: {output.shape}, dtype: {output.dtype}, contiguous: {output.is_contiguous()}")
        print("✓ Basic test passed!")
        
    except Exception as e:
        print(f"✗ Basic test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def test_depth_pro_batch():
    """Test DepthPro with batch input."""
    print("\nTesting DepthPro model with batch...")
    
    # Create model
    model = DepthProInference()
    
    # Create test input with batch
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    
    # Create random input tensor
    x = torch.randn(batch_size, channels, height, width)
    print(f"Batch input shape: {x.shape}, dtype: {x.dtype}, contiguous: {x.is_contiguous()}")
    
    try:
        # Test forward pass
        with torch.no_grad():
            output = model(x)
        print(f"Batch output shape: {output.shape}, dtype: {output.dtype}, contiguous: {output.is_contiguous()}")
        print("✓ Batch test passed!")
        
    except Exception as e:
        print(f"✗ Batch test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def test_depth_pro_different_sizes():
    """Test DepthPro with different input sizes."""
    print("\nTesting DepthPro model with different sizes...")
    
    # Create model
    model = DepthProInference()
    
    sizes = [(224, 224), (256, 256), (480, 640)]
    
    for height, width in sizes:
        print(f"\nTesting size: {height}x{width}")
        
        # Create test input
        x = torch.randn(1, 3, height, width)
        print(f"Input shape: {x.shape}, contiguous: {x.is_contiguous()}")
        
        try:
            # Test forward pass
            with torch.no_grad():
                output = model(x)
            print(f"Output shape: {output.shape}, contiguous: {output.is_contiguous()}")
            print(f"✓ Size {height}x{width} test passed!")
            
        except Exception as e:
            print(f"✗ Size {height}x{width} test failed: {str(e)}")


if __name__ == "__main__":
    test_depth_pro_basic()
    test_depth_pro_batch()
    test_depth_pro_different_sizes() 
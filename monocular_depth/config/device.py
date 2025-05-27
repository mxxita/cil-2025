"""Device configuration for the monocular depth estimation project."""

import torch
import os

# Disable MPS globally
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def get_device(prefer_cpu=False):
    """Get the best available device, excluding MPS.
    
    Args:
        prefer_cpu: If True, prefer CPU over GPU
        
    Returns:
        torch.device: The selected device (cuda or cpu only)
    """
    if prefer_cpu:
        device = torch.device("cpu")
        print(f"Device: {device} (CPU preferred)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {device} (CUDA available)")
    else:
        device = torch.device("cpu")
        print(f"Device: {device} (fallback to CPU)")
    
    return device

# Global device for the project
DEVICE = get_device()

print(f"Global device set to: {DEVICE}")
print("MPS is disabled for this project") 
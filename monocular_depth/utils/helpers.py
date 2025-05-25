import os
import torch
import yaml
from typing import Dict, Any, Optional

def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration file
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Get device for training.
    
    Args:
        device_name: Device name ('cuda' or 'cpu')
        
    Returns:
        torch.device object
    """
    if device_name is None:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device_name)
    
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Initially allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    else:
        print("Using CPU")
    
    return device

def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory after clearing: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB") 
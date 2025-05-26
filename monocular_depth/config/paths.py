"""Path configuration for different environments and clusters."""

import os
from pathlib import Path
from typing import Dict, Optional

# Base paths for different clusters
CLUSTER_PATHS = {
    'euler': {
        'base': '/cluster/home/mariberger/cil-2025',
        'data': '/cluster/scratch/mariberger/courses/cil/monocular_depth',
        'models': '/cluster/home/mariberger/cil-2025/monocular_depth/models',
        'outputs': '/cluster/home/mariberger/cil-2025/monocular_depth/outputs',
        'checkpoints': '/cluster/home/mariberger/cil-2025/checkpoints'
    },
    'local': {
        'base': str(Path(__file__).parent.parent.parent),
        'data': str(Path(__file__).parent.parent.parent / 'ml-depth-pro/data'),
        'models': str(Path(__file__).parent.parent.parent / 'ml-depth-pro/models'),
        'outputs': str(Path(__file__).parent.parent.parent / 'ml-depth-pro/outputs'),
        'checkpoints': str(Path(__file__).parent.parent.parent / 'checkpoints')
    }
}

# Current cluster detection
def detect_cluster() -> str:
    """
    Detect the current cluster/environment.
    
    Returns:
        str: Cluster name ('euler' or 'local')
    """
    if os.path.exists('/cluster'):
        return 'euler'
    return 'local'

# Global variables
CURRENT_CLUSTER = detect_cluster()
BASE_PATH = CLUSTER_PATHS[CURRENT_CLUSTER]['base']
DATA_PATH = CLUSTER_PATHS[CURRENT_CLUSTER]['data']
MODELS_PATH = CLUSTER_PATHS[CURRENT_CLUSTER]['models']
OUTPUTS_PATH = CLUSTER_PATHS[CURRENT_CLUSTER]['outputs']
CHECKPOINTS_PATH = CLUSTER_PATHS[CURRENT_CLUSTER]['checkpoints']

# Specific paths
TEST_IMAGE_PATH ='/cluster/home/mariberger/cil-2025/ml-depth-pro/data/example.jpg'

def get_path(path_type: str) -> str:
    """
    Get a specific path based on type.
    
    Args:
        path_type: Type of path ('base', 'data', 'models', 'outputs')
        
    Returns:
        str: Absolute path
    """
    if path_type not in CLUSTER_PATHS[CURRENT_CLUSTER]:
        raise ValueError(f"Unknown path type: {path_type}")
    return CLUSTER_PATHS[CURRENT_CLUSTER][path_type]

def get_test_image_path() -> str:
    """Get the path to the test image."""
    return TEST_IMAGE_PATH

def get_model_path(model_name: str) -> str:
    """
    Get the path to a specific model.
    
    Args:
        model_name: Name of the model ('depth_pro', 'unet', etc.)
        
    Returns:
        str: Absolute path to model directory
    """
    return os.path.join(MODELS_PATH, model_name)

def get_output_path(output_name: str) -> str:
    """
    Get the path to a specific output directory.
    
    Args:
        output_name: Name of the output directory
        
    Returns:
        str: Absolute path to output directory
    """
    return os.path.join(OUTPUTS_PATH, output_name)

def get_checkpoint_path(model_name: str) -> str:
    """
    Get the path to a model checkpoint.
    
    Args:
        model_name: Name of the model checkpoint (e.g., 'depth_pro.pt')
        
    Returns:
        str: Absolute path to checkpoint file
    """
    return os.path.join(CHECKPOINTS_PATH, model_name)

# Create directories if they don't exist
for path in [DATA_PATH, MODELS_PATH, OUTPUTS_PATH]:
    os.makedirs(path, exist_ok=True) 
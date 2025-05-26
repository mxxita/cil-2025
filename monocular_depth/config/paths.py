"""Path configuration for different environments and clusters."""

import os
from pathlib import Path
from typing import Dict, Optional

# Environment variable names
ENV_BASE_PATH = 'CIL_BASE_PATH'
ENV_DATA_PATH = 'CIL_DATA_PATH'
ENV_MODELS_PATH = 'CIL_MODELS_PATH'
ENV_OUTPUTS_PATH = 'CIL_OUTPUTS_PATH'
ENV_CHECKPOINTS_PATH = 'CIL_CHECKPOINTS_PATH'

# Default paths for different clusters
DEFAULT_PATHS = {
    'euler': {
        'base': '/cluster/home/mariberger/cil-2025',
        'data': '/cluster/scratch/mariberger/courses/cil/monocular_depth',
        'models': '/cluster/home/mariberger/cil-2025/ml-depth-pro/models',
        'outputs': '/cluster/home/mariberger/cil-2025/ml-depth-pro/outputs',
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

def get_env_path(env_var: str, default: str) -> str:
    """
    Get path from environment variable or use default.
    
    Args:
        env_var: Environment variable name
        default: Default path if environment variable is not set
        
    Returns:
        str: Path from environment variable or default
    """
    return os.getenv(env_var, default)

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
DEFAULT_CLUSTER_PATHS = DEFAULT_PATHS[CURRENT_CLUSTER]

# Get paths from environment variables or use defaults
BASE_PATH = get_env_path(ENV_BASE_PATH, DEFAULT_CLUSTER_PATHS['base'])
DATA_PATH = get_env_path(ENV_DATA_PATH, DEFAULT_CLUSTER_PATHS['data'])
MODELS_PATH = get_env_path(ENV_MODELS_PATH, DEFAULT_CLUSTER_PATHS['models'])
OUTPUTS_PATH = get_env_path(ENV_OUTPUTS_PATH, DEFAULT_CLUSTER_PATHS['outputs'])
CHECKPOINTS_PATH = get_env_path(ENV_CHECKPOINTS_PATH, DEFAULT_CLUSTER_PATHS['checkpoints'])

# Specific paths
TEST_IMAGE_PATH = os.path.join(DATA_PATH, 'example.jpg')

def get_path(path_type: str) -> str:
    """
    Get a specific path based on type.
    
    Args:
        path_type: Type of path ('base', 'data', 'models', 'outputs', 'checkpoints')
        
    Returns:
        str: Absolute path
    """
    if path_type not in DEFAULT_CLUSTER_PATHS:
        raise ValueError(f"Unknown path type: {path_type}")
    return get_env_path(f'CIL_{path_type.upper()}_PATH', DEFAULT_CLUSTER_PATHS[path_type])

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
"""Example usage of path configuration."""

import os
from monocular_depth.config.paths import (
    # Global path variables
    BASE_PATH,
    DATA_PATH,
    MODELS_PATH,
    OUTPUTS_PATH,
    CHECKPOINTS_PATH,
    # Helper functions
    get_path,
    get_test_image_path,
    get_model_path,
    get_output_path,
    get_checkpoint_path,
    # Cluster detection
    CURRENT_CLUSTER,
    detect_cluster
)

def example_path_usage():
    """Example of how to use the path configuration."""
    
    # 1. Print current environment
    print(f"Running on cluster: {CURRENT_CLUSTER}")
    
    # 2. Access global paths directly
    print("\nGlobal paths:")
    print(f"Base directory: {BASE_PATH}")
    print(f"Data directory: {DATA_PATH}")
    print(f"Models directory: {MODELS_PATH}")
    print(f"Outputs directory: {OUTPUTS_PATH}")
    print(f"Checkpoints directory: {CHECKPOINTS_PATH}")
    
    # 3. Get specific paths using helper functions
    print("\nSpecific paths:")
    test_image = get_test_image_path()
    print(f"Test image: {test_image}")
    
    model_dir = get_model_path('depth_pro')
    print(f"Model directory: {model_dir}")
    
    output_dir = get_output_path('predictions')
    print(f"Output directory: {output_dir}")
    
    checkpoint = get_checkpoint_path('depth_pro.pt')
    print(f"Checkpoint file: {checkpoint}")
    
    # 4. Get paths by type
    print("\nPaths by type:")
    data_path = get_path('data')
    print(f"Data path: {data_path}")
    
    # 5. Example of using paths in a typical workflow
    print("\nExample workflow:")
    
    # Load test image
    if os.path.exists(test_image):
        print(f"Found test image at: {test_image}")
    
    # Create output directory for predictions
    os.makedirs(get_output_path('predictions'), exist_ok=True)
    print("Created predictions directory")
    
    # Check if checkpoint exists
    if os.path.exists(checkpoint):
        print(f"Found model checkpoint at: {checkpoint}")
    else:
        print(f"Warning: Checkpoint not found at: {checkpoint}")

def example_with_depth_pro():
    """Example of using paths with DepthPro model."""
    from monocular_depth.models import DepthProInference
    from PIL import Image
    
    # Get paths
    test_image = get_test_image_path()
    output_dir = get_output_path('depth_maps')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model (it will use the checkpoint path internally)
    model = DepthProInference()
    
    # Process image and save output
    if os.path.exists(test_image):
        # Load and process image
        image = Image.open(test_image).convert('RGB')
        depth_map = model(image)
        
        # Save output
        output_path = os.path.join(output_dir, 'depth_map.png')
        model.visualize(depth_map, save_path=output_path)
        print(f"Saved depth map to: {output_path}")

if __name__ == '__main__':
    print("=== Basic Path Usage ===")
    example_path_usage()
    
    print("\n=== DepthPro Example ===")
    example_with_depth_pro() 
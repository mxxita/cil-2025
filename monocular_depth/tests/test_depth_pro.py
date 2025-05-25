import os
import pytest
import numpy as np
from PIL import Image
import torch
from monocular_depth.models import DepthProInference

def create_test_image(size=(256, 256)):
    """Create a simple test image."""
    # Create a gradient image
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
    X, Y = np.meshgrid(x, y)
    image = (X + Y) / 2
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

@pytest.fixture
def test_image_path(tmp_path):
    """Create a temporary test image file."""
    image = create_test_image()
    image_path = tmp_path / "test_image.jpg"
    image.save(image_path)
    return str(image_path)

@pytest.fixture
def depth_estimator():
    """Create a DepthProInference instance."""
    return DepthProInference(device='cpu')  # Use CPU for testing

def test_model_initialization(depth_estimator):
    """Test if the model initializes correctly."""
    assert depth_estimator.model is not None
    assert isinstance(depth_estimator.device, str)
    assert depth_estimator.model.training is False

def test_preprocess(depth_estimator, test_image_path):
    """Test image preprocessing."""
    # Test with file path
    tensor = depth_estimator.preprocess(test_image_path)
    assert isinstance(tensor, torch.Tensor)
    
    # Test with PIL Image
    image = Image.open(test_image_path)
    tensor = depth_estimator.preprocess(image)
    assert isinstance(tensor, torch.Tensor)
    
    # Test with numpy array
    array = np.array(image)
    tensor = depth_estimator.preprocess(array)
    assert isinstance(tensor, torch.Tensor)

def test_predict(depth_estimator, test_image_path):
    """Test depth prediction."""
    depth_map = depth_estimator.predict(test_image_path)
    assert isinstance(depth_map, np.ndarray)
    assert depth_map.ndim == 2  # Should be 2D (height, width)
    assert not np.isnan(depth_map).any()  # Should not contain NaN values
    assert not np.isinf(depth_map).any()  # Should not contain infinite values

def test_visualize(depth_estimator, test_image_path, tmp_path):
    """Test visualization functionality."""
    depth_map = depth_estimator.predict(test_image_path)
    
    # Test visualization without saving
    depth_estimator.visualize(depth_map)
    
    # Test visualization with saving
    save_path = tmp_path / "depth_map.png"
    depth_estimator.visualize(depth_map, save_path=str(save_path))
    assert save_path.exists()

def test_process_and_visualize(depth_estimator, test_image_path, tmp_path):
    """Test combined processing and visualization."""
    # Test without saving
    depth_map = depth_estimator.process_and_visualize(test_image_path)
    assert isinstance(depth_map, np.ndarray)
    
    # Test with saving
    save_path = tmp_path / "depth_map_combined.png"
    depth_map = depth_estimator.process_and_visualize(
        test_image_path, 
        save_path=str(save_path)
    )
    assert save_path.exists()
    assert isinstance(depth_map, np.ndarray)

if __name__ == '__main__':
    # Create a simple test image
    test_image = create_test_image()
    test_image_path = "test_image.jpg"
    test_image.save(test_image_path)
    
    try:
        # Initialize the model
        print("Initializing DepthPro model...")
        depth_estimator = DepthProInference(device='cpu')
        
        # Test the full pipeline
        print("Testing depth estimation pipeline...")
        depth_map = depth_estimator.process_and_visualize(
            test_image_path,
            save_path="test_depth_map.png"
        )
        print("Test completed successfully!")
        print(f"Depth map shape: {depth_map.shape}")
        print(f"Depth map range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        
    finally:
        # Cleanup
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        if os.path.exists("test_depth_map.png"):
            os.remove("test_depth_map.png") 
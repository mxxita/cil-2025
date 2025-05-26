"""Test DepthPro model inference."""

import os
import pytest
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from monocular_depth.models import DepthProInference
from monocular_depth.config.paths import get_test_image_path
import tempfile

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

def test_depth_pro_inference():
    """Test DepthPro model inference."""
    # Load test image
    image_path = get_test_image_path()
    image = Image.open(image_path).convert('RGB')
    
    # Create model
    model = DepthProInference()
    
    # Prepare input using DepthPro's exact transform
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.to(model.device)),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1] range
        T.ConvertImageDtype(torch.float32)
    ])
    x = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Check output
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 1, x.shape[2], x.shape[3])  # (batch_size, channels, height, width)
    assert torch.all(output >= 0) and torch.all(output <= 10)  # Depth in [0, 10] meters

if __name__ == '__main__':
    # Use the example image from ml-depth-pro
    example_image_path = os.path.join("ml-depth-pro", "data", "example.jpg")
    if not os.path.exists(example_image_path):
        raise FileNotFoundError(f"Example image not found at {example_image_path}")
    
    try:
        # Initialize the model
        print("Initializing DepthPro model...")
        depth_estimator = DepthProInference(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test the full pipeline
        print("Testing depth estimation pipeline...")
        depth_map = depth_estimator.process_and_visualize(
            example_image_path,
            save_path="test_depth_map.png"
        )
        print("Test completed successfully!")
        print(f"Depth map shape: {depth_map.shape}")
        print(f"Depth map range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise 
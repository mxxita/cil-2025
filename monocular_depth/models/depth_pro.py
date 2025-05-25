import torch
from PIL import Image
import numpy as np
from depth_pro import DepthProModel
from typing import Union, Tuple
import matplotlib.pyplot as plt

class DepthProInference:
    """Wrapper class for DepthPro model inference."""
    
    def __init__(self, model_name: str = 'depth-pro', device: str = None):
        """
        Initialize the DepthPro model.
        
        Args:
            model_name: Name of the pretrained model to load
            device: Device to run inference on ('cuda' or 'cpu'). If None, will use CUDA if available.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Load the pretrained model
        self.model = DepthProModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
    
    def preprocess(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess an image for depth estimation.
        
        Args:
            image: Input image as file path, PIL Image, or numpy array
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        return self.model.preprocess(image)
    
    def predict(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Perform depth estimation on an image.
        
        Args:
            image: Input image as file path, PIL Image, or numpy array
            
        Returns:
            Depth map as numpy array
        """
        # Preprocess the image
        input_tensor = self.preprocess(image)
        input_tensor = input_tensor.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            depth_map = self.model(input_tensor)
            
        # Convert to numpy array
        depth_map = depth_map.cpu().numpy()
        
        # Remove batch dimension if present
        if depth_map.ndim == 4:
            depth_map = depth_map[0]
        if depth_map.ndim == 3:
            depth_map = depth_map[0]
            
        return depth_map
    
    def visualize(self, depth_map: np.ndarray, 
                 figsize: Tuple[int, int] = (10, 8),
                 cmap: str = 'magma',
                 save_path: str = None) -> None:
        """
        Visualize the depth map.
        
        Args:
            depth_map: Depth map as numpy array
            figsize: Figure size for visualization
            cmap: Colormap to use
            save_path: If provided, save the visualization to this path
        """
        plt.figure(figsize=figsize)
        plt.imshow(depth_map, cmap=cmap)
        plt.colorbar(label='Depth')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()
    
    def process_and_visualize(self, image: Union[str, Image.Image, np.ndarray],
                            save_path: str = None) -> np.ndarray:
        """
        Process an image and visualize the depth map in one step.
        
        Args:
            image: Input image as file path, PIL Image, or numpy array
            save_path: If provided, save the visualization to this path
            
        Returns:
            Depth map as numpy array
        """
        depth_map = self.predict(image)
        self.visualize(depth_map, save_path=save_path)
        return depth_map 
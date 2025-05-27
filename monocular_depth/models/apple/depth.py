import torch
import torch.nn as nn
from typing import Tuple

from depth_pro import create_model_and_transforms
from monocular_depth.config.device import get_device


class DepthProInference(nn.Module):
    """DepthPro with frozen backbone + trainable MLP head for depth refinement."""
    
    def __init__(self, prefer_cpu=True, enable_training=True):
        """Initialize the DepthPro model with optional trainable MLP head.
        
        Args:
            prefer_cpu: If True, use CPU to avoid memory issues
            enable_training: If True, add trainable MLP head for depth refinement
        """
        super().__init__()
        
        # Load frozen DepthPro backbone
        self.backbone, self.transform = create_model_and_transforms()
        self.enable_training = enable_training
        
        # Use centralized device configuration (no MPS)
        self.device = get_device(prefer_cpu=prefer_cpu)
        self.backbone = self.backbone.to(self.device)
        
        # Always freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        if enable_training:
            # Add trainable MLP head for depth refinement
            # Takes depth map as input and outputs refined depth map
            self.depth_mlp = nn.Sequential(
                # Pointwise convolution acts as MLP for each spatial location
                nn.Conv2d(1, 64, kernel_size=1),  # 1x1 conv = pointwise MLP
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1),  # Output refined depth
                nn.ReLU(inplace=True)  # Ensure positive depth values
            ).to(self.device)
            
            print(f"DepthPro initialized on device: {self.device}")
            print(f"Backbone: FROZEN (inference-only)")
            print(f"MLP head: TRAINABLE ({sum(p.numel() for p in self.depth_mlp.parameters())} parameters)")
        else:
            self.depth_mlp = None
            print(f"DepthPro initialized on device: {self.device} (inference-only)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, 1, height, width)
        """
        # Store original device
        original_device = x.device
        
        # Move input to model device
        x = x.contiguous().to(self.device)

        outputs = []
        try:
            # Extract depth features using frozen backbone
            with torch.no_grad():
                prediction = self.backbone.infer(x)
                depth_map = prediction["depth"]  # Shape can be (height, width) or (batch_size, height, width)
                depth_map = depth_map.contiguous()
                
                # Ensure proper batch and channel dimensions
                if depth_map.dim() == 2:
                    # Shape: (height, width) -> (1, 1, height, width)
                    depth_map = depth_map.unsqueeze(0).unsqueeze(0)
                elif depth_map.dim() == 3:
                    # Shape: (batch_size, height, width) -> (batch_size, 1, height, width)
                    depth_map = depth_map.unsqueeze(1)
                elif depth_map.dim() == 4:
                    # Already correct shape: (batch_size, 1, height, width)
                    pass
                else:
                    raise ValueError(f"Unexpected depth map dimensions: {depth_map.shape}")
                
                print(f"DepthPro backbone output - Shape: {depth_map.shape}, Device: {depth_map.device}")

            # Apply trainable MLP head if enabled
            if self.enable_training and self.depth_mlp is not None:
                # Apply MLP refinement (this has gradients)
                refined_depth = self.depth_mlp(depth_map)
                print(f"MLP refined output - Shape: {refined_depth.shape}, Device: {refined_depth.device}")
                # Output should be (batch_size, 1, height, width) to match target format
                output = refined_depth
            else:
                # Just convert raw depth to disparity for inference
                # Remove channel dimension to match target format (batch_size, height, width)
                depth = 1.0 / (depth_map.squeeze(1) + 1e-6)
                depth = torch.clamp(depth, 0.0, 10.0)
                # Add channel dimension back: (batch_size, height, width) -> (batch_size, 1, height, width)
                output = depth.unsqueeze(1)
                
            # Move result back to original device if needed
            if original_device != self.device:
                output = output.to(original_device)

            return output

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n✗ Batch processing failed due to OOM. Falling back to single-image processing...\n")
                for i in range(x.shape[0]):
                    x_single = x[i:i+1].contiguous()
                    with torch.no_grad():
                        prediction = self.backbone.infer(x_single)
                        depth_single = prediction["depth"].contiguous()
                        if depth_single.dim() == 2:
                            depth_single = depth_single.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                        elif depth_single.dim() == 3:
                            depth_single = depth_single.unsqueeze(1)  # Add channel dim
                    
                    # Apply MLP if enabled
                    if self.enable_training and self.depth_mlp is not None:
                        refined_depth = self.depth_mlp(depth_single)
                        output_single = refined_depth
                    else:
                        depth = 1.0 / (depth_single.squeeze(1) + 1e-6)
                        depth = torch.clamp(depth, 0.0, 10.0)
                        output_single = depth.unsqueeze(1)
                    
                    # Move to original device if needed
                    if original_device != self.device:
                        output_single = output_single.to(original_device)
                    
                    outputs.append(output_single)
                    print(f"  ✓ Image {i+1} output: {output_single.shape}")

                final_output = torch.cat(outputs, dim=0)
                print(f"✓ Individual processing SUCCESS - Final shape: {final_output.shape}")
                return final_output
            else:
                raise e

    def train(self, mode=True):
        """Override train method - only MLP head should be trainable."""
        super().train(mode)
        # Always keep backbone frozen
        if hasattr(self, 'backbone'):
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        # MLP head follows training mode
        if hasattr(self, 'depth_mlp') and self.depth_mlp is not None:
            self.depth_mlp.train(mode)
        return self

    def eval(self):
        """Override eval method."""
        super().eval()
        if hasattr(self, 'backbone'):
            self.backbone.eval()
        if hasattr(self, 'depth_mlp') and self.depth_mlp is not None:
            self.depth_mlp.eval()
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward pass to match UNet interface."""
        return self.forward(x)

    def _patch_model_for_contiguity(self):
        """Patch the model to handle tensor contiguity issues."""
        # This is a workaround for the ml-depth-pro package's tensor memory layout issues
        original_forward_methods = {}
        
        def make_contiguous_wrapper(original_method):
            def wrapper(*args, **kwargs):
                # Make all tensor arguments contiguous
                new_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        new_args.append(arg.contiguous())
                    else:
                        new_args.append(arg)
                
                new_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        new_kwargs[k] = v.contiguous()
                    else:
                        new_kwargs[k] = v
                
                try:
                    result = original_method(*new_args, **new_kwargs)
                    # Make result contiguous if it's a tensor
                    if isinstance(result, torch.Tensor):
                        return result.contiguous()
                    elif isinstance(result, (list, tuple)):
                        return type(result)(
                            item.contiguous() if isinstance(item, torch.Tensor) else item
                            for item in result
                        )
                    return result
                except RuntimeError as e:
                    if "view size is not compatible" in str(e):
                        print(f"Caught view error in {original_method.__name__}, trying with contiguous tensors...")
                        # Try again with all tensors made contiguous
                        new_args = [arg.contiguous() if isinstance(arg, torch.Tensor) else arg for arg in new_args]
                        new_kwargs = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in new_kwargs.items()}
                        result = original_method(*new_args, **new_kwargs)
                        if isinstance(result, torch.Tensor):
                            return result.contiguous()
                        return result
                    else:
                        raise
            return wrapper
        
        # Patch all modules in the model
        for name, module in self.backbone.named_modules():
            if hasattr(module, 'forward'):
                original_forward_methods[name] = module.forward
                module.forward = make_contiguous_wrapper(module.forward)

        
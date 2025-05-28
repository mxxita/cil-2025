import torch
import torch.nn as nn
from typing import Tuple

from depth_pro import create_model_and_transforms
from monocular_depth.config.device import get_device
from monocular_depth.models.apple.encoder_new import CustomFSCNFusionStage, AdaptiveConcatenationModule, ChannelAttentionModule


class DepthProWithFSCN(nn.Module):
    """
    Enhanced DepthPro with FSCN (Feature Spatial Concatenation Network) decoder.
    
    This model combines the frozen DepthPro backbone with an advanced FSCN decoder
    that uses attention mechanisms, adaptive concatenation, and edge-preserving
    convolutions for superior depth estimation.
    
    Architecture:
        1. Frozen DepthPro backbone (for feature extraction)
        2. Feature extraction at multiple scales (5 scales) from the backbone
        3. FSCN-based fusion with attention mechanisms
        4. Progressive upsampling and refinement
    
    Key Improvements over standard DepthPro:
        - Multi-scale feature fusion with attention (5 scales)
        - Edge-preserving convolutions using Sobel filters
        - Adaptive concatenation of features from different scales
        - Channel and spatial attention mechanisms (CBAM-style)
    """
    
    def __init__(self, prefer_cpu=True, enable_training=True):
        """
        Initialize DepthPro with FSCN decoder.
        
        Args:
            prefer_cpu: If True, use CPU to avoid memory issues
            enable_training: If True, enable training mode for FSCN components
        """
        super().__init__()
        
        # Load frozen DepthPro backbone
        self.backbone, self.transform = create_model_and_transforms()
        self.enable_training = enable_training
        
        # Use centralized device configuration
        self.device = get_device(prefer_cpu=prefer_cpu)
        self.backbone = self.backbone.to(self.device)
        
        # Always freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        if enable_training:
            # Create FSCN decoder components
            # Note: We'll need to adapt this to work with DepthPro's output structure
            
            # Multi-scale feature extraction from depth map
            self.feature_extractors = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2**i) if i > 0 else nn.Identity()
                ) for i in range(5)  # 5 different scales
            ]).to(self.device)
            
            # Adaptive Concatenation Module for fusing multi-scale features
            self.fusion_module = AdaptiveConcatenationModule(
                in_channels=128, 
                num_stages=5  # Updated to 5 stages
            ).to(self.device)
            
            # Progressive refinement layers with attention
            self.refinement_layers = nn.ModuleList([
                nn.Sequential(
                    ChannelAttentionModule(128),
                    nn.Conv2d(128, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(32, 16, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 8, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(8, 4, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(4, 1, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)  # Ensure positive depth values
                )
            ]).to(self.device)
            
            # Edge enhancement module
            self.edge_enhancer = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),
                nn.Sigmoid()  # Edge attention weights
            ).to(self.device)
            
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"DepthPro+FSCN initialized on device: {self.device}")
            print(f"Backbone: FROZEN (inference-only)")
            print(f"FSCN Decoder: TRAINABLE ({total_params} parameters)")
        else:
            self.feature_extractors = None
            self.fusion_module = None
            self.refinement_layers = None
            self.edge_enhancer = None
            print(f"DepthPro+FSCN initialized on device: {self.device} (inference-only)")

    def extract_multi_scale_features(self, depth_map):
        """
        Extract features at multiple scales from the depth map.
        
        Args:
            depth_map: Tensor of shape (B, 1, H, W)
            
        Returns:
            List of feature tensors at different scales
        """
        features = []
        for extractor in self.feature_extractors:
            feature = extractor(depth_map)
            features.append(feature)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FSCN enhancement.
        
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
            # Extract initial depth using frozen backbone
            with torch.no_grad():
                prediction = self.backbone.infer(x)
                depth_map = prediction["depth"]
                depth_map = depth_map.contiguous()
                
                # Ensure proper batch and channel dimensions
                if depth_map.dim() == 2:
                    depth_map = depth_map.unsqueeze(0).unsqueeze(0)
                elif depth_map.dim() == 3:
                    depth_map = depth_map.unsqueeze(1)
                elif depth_map.dim() == 4:
                    pass
                else:
                    raise ValueError(f"Unexpected depth map dimensions: {depth_map.shape}")
                
                print(f"DepthPro backbone output - Shape: {depth_map.shape}, Device: {depth_map.device}")

            # Apply FSCN refinement if enabled
            if self.enable_training and self.fusion_module is not None:
                # Step 1: Extract multi-scale features from initial depth
                multi_scale_features = self.extract_multi_scale_features(depth_map)
                print(f"Extracted {len(multi_scale_features)} multi-scale features")
                
                # Step 2: Fuse features using FSCN
                # Use the largest feature as the decoder feature
                decoder_feature = multi_scale_features[0]  # Full resolution feature
                encoder_features = multi_scale_features[1:]  # Downsampled features
                
                fused_feature = self.fusion_module(decoder_feature, encoder_features)
                print(f"FSCN fused feature - Shape: {fused_feature.shape}")
                
                # Step 3: Progressive refinement
                refined_feature = fused_feature
                for i, layer in enumerate(self.refinement_layers):
                    refined_feature = layer(refined_feature)
                    # Upsample if not the last layer
                    if i < len(self.refinement_layers) - 1:
                        refined_feature = nn.functional.interpolate(
                            refined_feature, 
                            scale_factor=2, 
                            mode="bilinear", 
                            align_corners=False
                        )
                    print(f"Refinement layer {i+1} output - Shape: {refined_feature.shape}")
                
                # Step 4: Edge enhancement
                edge_weights = self.edge_enhancer(refined_feature)
                
                # Step 5: Combine original depth with refined depth using edge weights
                # Resize refined depth to match original depth
                if refined_feature.shape != depth_map.shape:
                    refined_feature = nn.functional.interpolate(
                        refined_feature, 
                        size=depth_map.shape[2:], 
                        mode="bilinear", 
                        align_corners=False
                    )
                    edge_weights = nn.functional.interpolate(
                        edge_weights, 
                        size=depth_map.shape[2:], 
                        mode="bilinear", 
                        align_corners=False
                    )
                
                # Final fusion with edge-aware blending
                output = edge_weights * refined_feature + (1 - edge_weights) * depth_map
                print(f"FSCN enhanced output - Shape: {output.shape}")
                
            else:
                # Inference mode: just convert depth to disparity
                depth = 1.0 / (depth_map.squeeze(1) + 1e-6)
                depth = torch.clamp(depth, 0.0, 10.0)
                output = depth.unsqueeze(1)
            
            # Move result back to original device if needed
            if original_device != self.device:
                output = output.to(original_device)

            return output

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n✗ Batch processing failed due to OOM. Falling back to single-image processing...\n")
                # Implement single-image fallback similar to original
                for i in range(x.shape[0]):
                    x_single = x[i:i+1].contiguous()
                    # Process single image (simplified for OOM case)
                    with torch.no_grad():
                        prediction = self.backbone.infer(x_single)
                        depth_single = prediction["depth"].contiguous()
                        if depth_single.dim() == 2:
                            depth_single = depth_single.unsqueeze(0).unsqueeze(0)
                        elif depth_single.dim() == 3:
                            depth_single = depth_single.unsqueeze(1)
                    
                    # For OOM, skip FSCN and use simple processing
                    depth = 1.0 / (depth_single.squeeze(1) + 1e-6)
                    depth = torch.clamp(depth, 0.0, 10.0)
                    output_single = depth.unsqueeze(1)
                    
                    if original_device != self.device:
                        output_single = output_single.to(original_device)
                    
                    outputs.append(output_single)

                final_output = torch.cat(outputs, dim=0)
                print(f"✓ Individual processing SUCCESS - Final shape: {final_output.shape}")
                return final_output
            else:
                raise e

    def train(self, mode=True):
        """Override train method - only FSCN components should be trainable."""
        super().train(mode)
        # Always keep backbone frozen
        if hasattr(self, 'backbone'):
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # FSCN components follow training mode
        trainable_modules = ['feature_extractors', 'fusion_module', 'refinement_layers', 'edge_enhancer']
        for module_name in trainable_modules:
            if hasattr(self, module_name) and getattr(self, module_name) is not None:
                getattr(self, module_name).train(mode)
        return self

    def eval(self):
        """Override eval method."""
        super().eval()
        if hasattr(self, 'backbone'):
            self.backbone.eval()
        
        trainable_modules = ['feature_extractors', 'fusion_module', 'refinement_layers', 'edge_enhancer']
        for module_name in trainable_modules:
            if hasattr(self, module_name) and getattr(self, module_name) is not None:
                getattr(self, module_name).eval()
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward pass to match UNet interface."""
        return self.forward(x)


# Keep the original class for backward compatibility
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

        
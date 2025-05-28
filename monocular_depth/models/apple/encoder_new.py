"""
Custom Feature Spatial Concatenation Network (FSCN) Fusion Stage

This module implements a custom fusion mechanism for Dense Prediction Transformer (DPT) models
that replaces the standard DPTFeatureFusionStage with an advanced Feature Spatial Concatenation
Network (FSCN) approach. The FSCN fusion incorporates spatial and channel attention mechanisms,
adaptive concatenation, and edge-preserving convolutions to improve feature fusion quality.

Key Components:
- ChannelAttentionModule: Implements channel-wise attention similar to SENet
- SpatialAttentionModule: Applies spatial attention weights
- AdaptiveConcatenationModule: Adaptively fuses multi-scale features with attention
- CustomFeatureFusionLayer: Enhanced fusion layer with FSCN integration
- CustomFSCNFusionStage: Main fusion stage replacing DPT's standard fusion

Architecture Overview:
The FSCN fusion stage processes features from multiple scales of the Vision Transformer
backbone and progressively fuses them using attention mechanisms and adaptive concatenation.
This approach aims to preserve fine-grained spatial details while maintaining global context.

Usage:
    from models.dpt_hybrid_midas.fscn.CustomFSCNFusionStage import CustomFSCNFusionStage
    
    # Replace standard DPT fusion stage
    fusion_stage = CustomFSCNFusionStage(config)
    fused_features = fusion_stage(encoder_features)

References:
    - Dense Prediction Transformer: https://arxiv.org/abs/2103.13413
    - CBAM: Convolutional Block Attention Module: https://arxiv.org/abs/1807.06521
    - Squeeze-and-Excitation Networks: https://arxiv.org/abs/1709.01507
"""

import torch
import torch.nn as nn
from transformers.models.dpt.modeling_dpt import DPTFeatureFusionLayer, DPTFeatureFusionStage, DPTForDepthEstimation


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg = self.avg_pool(x).view(x.size(0), -1)
        weights = self.mlp(avg).view(x.size(0), x.size(1), 1, 1)
        return x * weights

class CustomFeatureFusionLayer(DPTFeatureFusionLayer):
    def __init__(self, config, align_corners=True):
        super().__init__(config, align_corners)
        self.attention = ChannelAttentionModule(in_channels=config.fusion_hidden_size)

    def forward(self, hidden_state, residual=None):
        hidden_state = self.attention(hidden_state)
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual, size=(hidden_state.shape[2], hidden_state.shape[3]), 
                    mode="bilinear", align_corners=False
                )
            hidden_state = hidden_state + self.residual_layer1(residual)
        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = nn.functional.interpolate(
            hidden_state, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        hidden_state = self.projection(hidden_state)
        return hidden_state

class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module (CAM) inspired by SENet and CBAM.
    
    This module applies channel-wise attention by computing channel statistics
    through global average and max pooling, then generating attention weights
    via a multi-layer perceptron (MLP).
    
    Args:
        in_channels (int): Number of input channels
        reduction (int, optional): Channel reduction ratio for MLP. Defaults to 16.
        
    Shape:
        - Input: (batch_size, in_channels, height, width)
        - Output: (batch_size, in_channels, height, width)
        
    Example:
        >>> cam = ChannelAttentionModule(256, reduction=16)
        >>> x = torch.randn(2, 256, 32, 32)
        >>> out = cam(x)  # Same shape as input but with channel attention applied
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Global pooling layers to capture channel statistics
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling
        
        # MLP for generating channel attention weights
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()  # Output attention weights in [0, 1]
        )

    def forward(self, x):
        """
        Forward pass of Channel Attention Module.
        
        Args:
            x (torch.Tensor): Input feature tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Channel-attended features of shape (B, C, H, W)
        """
        # Compute global statistics
        avg = self.avg_pool(x).view(x.size(0), -1)  # (B, C)
        max_out = self.max_pool(x).view(x.size(0), -1)  # (B, C)
        
        # Generate attention weights by combining avg and max statistics
        weights = self.mlp(avg) + self.mlp(max_out)  # (B, C)
        weights = weights.view(x.size(0), x.size(1), 1, 1)  # (B, C, 1, 1)
        
        # Apply channel attention
        return x * weights


class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module (SAM) from CBAM.
    
    This module generates spatial attention weights by analyzing the spatial
    relationships in feature maps using channel-wise statistics (average and max).
    
    Shape:
        - Input: (batch_size, channels, height, width)
        - Output: (batch_size, 1, height, width) - spatial attention map
        
    Example:
        >>> sam = SpatialAttentionModule()
        >>> x = torch.randn(2, 256, 32, 32)
        >>> attention_map = sam(x)  # Shape: (2, 1, 32, 32)
    """
    def __init__(self):
        super().__init__()
        # Convolution to process concatenated channel statistics
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of Spatial Attention Module.
        
        Args:
            x (torch.Tensor): Input feature tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Spatial attention map of shape (B, 1, H, W)
        """
        # Compute channel-wise statistics across spatial dimensions
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate statistics and generate spatial attention
        x = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        x = self.conv(x)  # (B, 1, H, W)
        return self.sigmoid(x)


class AdaptiveConcatenationModule(nn.Module):
    """
    Adaptive Concatenation Module (ACM) for multi-scale feature fusion.
    
    This is the core component of the FSCN approach. It adaptively fuses features
    from multiple encoder stages with the current decoder feature using:
    1. Spatial attention for each encoder feature
    2. Adaptive weighting of encoder features
    3. Channel and spatial attention (CBAM-style)
    4. Edge-preserving convolutions for detail preservation
    
    Args:
        in_channels (int): Number of channels in decoder features
        num_stages (int): Number of encoder stages to fuse (typically 4 for DPT)
        
    Architecture Flow:
        encoder_features → spatial_attention → adaptive_weighting → 
        concatenation → CBAM_attention → edge_preservation → output
        
    Example:
        >>> acm = AdaptiveConcatenationModule(256, num_stages=4)
        >>> decoder_feat = torch.randn(2, 256, 32, 32)
        >>> encoder_feats = [torch.randn(2, 256, s, s) for s in [8, 16, 32, 64]]
        >>> fused = acm(decoder_feat, encoder_feats)
    """
    def __init__(self, in_channels, num_stages):
        super().__init__()
        self.num_stages = num_stages
        total_in_channels = in_channels * (num_stages + 1)  # +1 for decoder feature

        # Adaptive weighting network for encoder features
        self.adaptive_weights = nn.Sequential(
            nn.Conv2d(total_in_channels, num_stages, kernel_size=1, bias=False),
            nn.Sigmoid()  # Weights in [0, 1]
        )
        
        # Attention modules (CBAM-style)
        self.channel_attention = ChannelAttentionModule(total_in_channels)
        self.spatial_attention = SpatialAttentionModule()

        # Intermediate processing with dilated convolution for larger receptive field
        self.intermediate_conv = nn.Sequential(
            nn.Conv2d(total_in_channels, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU()
        )

        # Edge-preserving convolution using Sobel filters
        self.edge_preservation = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self._initialize_sobel_filters()
        
        # Final output projection
        self.conv = nn.Conv2d(512, in_channels, kernel_size=1)

    def _initialize_sobel_filters(self):
        """
        Initialize edge-preserving convolution with Sobel filters.
        
        Sobel filters are used to detect edges in horizontal and vertical directions,
        helping preserve important structural details during fusion.
        """
        with torch.no_grad():
            # Sobel filter kernels for edge detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            
            # Assign Sobel filters to weight matrix (alternating x and y filters)
            for i in range(512):
                self.edge_preservation.weight[i, i, :, :] = sobel_x if i % 2 == 0 else sobel_y
        
        self.conv = nn.Conv2d(512, in_channels, kernel_size=1)

    def forward(self, decoder_feature, encoder_features):
        """
        Forward pass of Adaptive Concatenation Module.
        
        Args:
            decoder_feature (torch.Tensor): Current decoder feature (B, C, H, W)
            encoder_features (List[torch.Tensor]): List of encoder features from different stages
            
        Returns:
            torch.Tensor: Fused feature tensor of shape (B, C, H, W)
            
        Process:
            1. Upsample encoder features to match decoder resolution
            2. Apply spatial attention to each encoder feature
            3. Compute adaptive weights for encoder features
            4. Concatenate and apply CBAM attention
            5. Apply edge-preserving convolution
            6. Return fused features
        """
        # Step 1: Upsample encoder features and apply spatial attention
        upsampled_features = []
        for feature in encoder_features:
            # Upsample to match decoder feature resolution
            feature = nn.functional.interpolate(
                feature, size=decoder_feature.shape[2:], mode="bicubic", align_corners=False
            )
            # Apply spatial attention to focus on important spatial regions
            spatial_weight = self.spatial_attention(feature)
            feature = feature * spatial_weight
            upsampled_features.append(feature)
        
        # Step 2: Compute adaptive weights for encoder features
        concat_features = torch.cat(upsampled_features + [decoder_feature], dim=1)
        weights = self.adaptive_weights(concat_features)  # (B, num_stages, H, W)
        weights = torch.softmax(weights, dim=1)  # Normalize weights
        weights = weights.split(1, dim=1)  # Split into individual weight maps
        
        # Step 3: Apply adaptive weights to encoder features
        weighted_features = [w * f for w, f in zip(weights, upsampled_features)]
        concat_features = torch.cat(weighted_features + [decoder_feature], dim=1)
        
        # CBAM-like attention
        channel_features = self.channel_attention(concat_features)
        channel_features = concat_features + channel_features  # Residual connection
        
        spatial_weights = self.spatial_attention(channel_features)
        fused_features = channel_features * spatial_weights
        fused_features = channel_features + fused_features  # Residual connection
        
        # Step 5: Intermediate processing with dilated convolution
        fused_features = self.intermediate_conv(fused_features)

        # Step 6: Edge preservation using Sobel filters
        fused_features = fused_features + self.edge_preservation(fused_features)

        # Step 7: Final projection and activation
        return nn.ReLU()(self.conv(fused_features))


class CustomFeatureFusionLayer(DPTFeatureFusionLayer):
    """
    Custom Feature Fusion Layer that replaces DPT's standard fusion with FSCN.
    
    This layer extends the standard DPTFeatureFusionLayer by incorporating the
    AdaptiveConcatenationModule (ACM) for enhanced multi-scale feature fusion.
    
    Args:
        config: DPT configuration object containing fusion parameters
        align_corners (bool, optional): Whether to align corners in interpolation. Defaults to True.
        
    Key Differences from Standard DPTFeatureFusionLayer:
        - Uses ACM instead of simple concatenation
        - Incorporates attention mechanisms
        - Preserves edge information through specialized convolutions
        
    Example:
        >>> layer = CustomFeatureFusionLayer(config)
        >>> decoder_state = torch.randn(2, 256, 32, 32)
        >>> encoder_features = [torch.randn(2, 256, s, s) for s in [8, 16, 32, 64]]
        >>> output = layer(decoder_state, encoder_features)
    """
    def __init__(self, config, align_corners=True):
        super().__init__(config, align_corners)
        # Replace standard fusion with Adaptive Concatenation Module
        self.acm = AdaptiveConcatenationModule(
            in_channels=config.fusion_hidden_size, 
            num_stages=4  # 4 encoder stages from DPT backbone (layers 3, 6, 9, 12)
        )

    def forward(self, hidden_state, encoder_features):
        """
        Forward pass with FSCN fusion.
        
        Args:
            hidden_state (torch.Tensor): Current decoder hidden state (B, C, H, W)
            encoder_features (List[torch.Tensor]): Encoder features from multiple stages
            
        Returns:
            torch.Tensor: Fused and upsampled feature tensor
            
        Process:
            1. Apply ACM to fuse encoder features with decoder state
            2. Apply residual layer and upsampling (from parent class)
            3. Apply final projection
        """
        # Step 1: Use ACM to fuse all encoder features with current decoder feature
        hidden_state = self.acm(hidden_state, encoder_features)

        # Step 2: Apply standard DPT processing (residual + upsampling + projection)
        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = nn.functional.interpolate(
            hidden_state, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        hidden_state = self.projection(hidden_state)
        return hidden_state


class CustomFSCNFusionStage(DPTFeatureFusionStage):
    """
    Custom Feature Spatial Concatenation Network (FSCN) Fusion Stage.
    
    This class replaces the standard DPTFeatureFusionStage with an enhanced fusion
    mechanism that uses Feature Spatial Concatenation Network (FSCN) approach.
    The FSCN incorporates spatial and channel attention, adaptive concatenation,
    and edge-preserving convolutions for improved feature fusion.
    
    Key Features:
        - Multi-scale feature fusion with attention mechanisms
        - Adaptive weighting of encoder features at each fusion layer
        - Edge-preserving convolutions using Sobel filters
        - CBAM-style channel and spatial attention
        - Progressive fusion through multiple layers
    
    Architecture:
        The fusion stage consists of 4 CustomFeatureFusionLayers, each processing
        features from the Vision Transformer backbone in a coarse-to-fine manner.
        Each layer receives all encoder features and adaptively fuses them.
    
    Args:
        config: DPT configuration object with fusion parameters
        
    Input:
        features (List[torch.Tensor]): List of 4 feature tensors from DPT backbone
            - features[0]: From layer 3 (finest scale)
            - features[1]: From layer 6  
            - features[2]: From layer 9
            - features[3]: From layer 12 (coarsest scale)
            
    Output:
        List[torch.Tensor]: List of 4 progressively fused feature tensors
        
    Usage Example:
        ```python
        # In your DPT model
        fusion_stage = CustomFSCNFusionStage(config)
        
        # Replace the standard fusion stage
        model.neck.fusion_stage = fusion_stage
        
        # Forward pass
        encoder_features = backbone(pixel_values)  # List of 4 tensors
        fused_features = fusion_stage(encoder_features)
        ```
        
    Mathematical Formulation:
        For each fusion layer i:
        F_i = FSCN(H_{i-1}, [E_1, E_2, E_3, E_4])
        
        Where:
        - F_i: Fused feature at layer i
        - H_{i-1}: Previous decoder hidden state (or last encoder feature for i=0)
        - E_j: Encoder feature from stage j
        - FSCN: Feature Spatial Concatenation Network (ACM + attention)
        
    Performance Benefits:
        - Better preservation of fine-grained spatial details
        - Improved handling of multi-scale information
        - Enhanced edge preservation through Sobel filtering
        - Adaptive attention-based feature weighting
        
    References:
        - Dense Prediction Transformer: https://arxiv.org/abs/2103.13413
        - CBAM: Convolutional Block Attention Module: https://arxiv.org/abs/1807.06521
    """
    
    def __init__(self, config):
        """
        Initialize the Custom FSCN Fusion Stage.
        
        Args:
            config: DPT configuration object containing:
                - fusion_hidden_size: Number of channels for fusion
                - Other DPT-specific parameters
        """
        super().__init__(config)
        # Replace standard fusion layers with custom FSCN layers
        self.layers = nn.ModuleList([
            CustomFeatureFusionLayer(config) for _ in range(4)
        ])

    def forward(self, features):
        """
        Forward pass of the FSCN Fusion Stage.
        
        Args:
            features (List[torch.Tensor]): List of 4 encoder features from DPT backbone
                - Ordered from finest to coarsest scale
                - Each tensor shape: (batch_size, hidden_size, height_i, width_i)
                
        Returns:
            List[torch.Tensor]: List of 4 progressively fused feature tensors
                - Each tensor represents output from corresponding fusion layer
                - Progressively upsampled and refined features
                
        Process:
            1. Reverse feature order (coarse-to-fine processing)
            2. Initialize with coarsest feature
            3. Progressively fuse with all encoder features at each layer
            4. Each layer receives both current decoder state and all encoder features
            5. Return list of all intermediate fused states
            
        Note:
            Unlike standard DPT fusion which only uses adjacent features,
            FSCN passes ALL encoder features to each fusion layer for
            comprehensive multi-scale information integration.
        """
        # Reverse features for coarse-to-fine processing
        hidden_states = features[::-1]  # [coarsest, ..., finest]

        fused_hidden_states = []
        fused_hidden_state = None
        
        # Progressive fusion through 4 layers
        for hidden_state, layer in zip(hidden_states, self.layers):
            if fused_hidden_state is None:
                # First layer: initialize with coarsest encoder feature
                fused_hidden_state = hidden_state
            
            # FSCN Innovation: Pass ALL encoder features to each layer
            # This allows each fusion layer to adaptively select and combine
            # information from all scales, not just adjacent ones
            fused_hidden_state = layer(fused_hidden_state, features)
            fused_hidden_states.append(fused_hidden_state)
            
        return fused_hidden_states
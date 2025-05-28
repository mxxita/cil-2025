# DepthPro + FSCN Architecture Summary

## Enhanced DepthProWithFSCN Model

### Overview
The DepthProWithFSCN model combines a frozen DepthPro backbone with an advanced Feature Spatial Concatenation Network (FSCN) decoder for superior depth estimation.

### Architecture Components

#### 1. Frozen DepthPro Backbone
- **Purpose**: Initial depth estimation
- **Status**: Frozen (no gradients)
- **Output**: Raw depth map of shape (B, 1, H, W)

#### 2. Multi-Scale Feature Extraction (5 Scales)
```python
Scale 0: Full resolution     (no pooling)
Scale 1: 1/2 resolution     (MaxPool2d(2))
Scale 2: 1/4 resolution     (MaxPool2d(4))
Scale 3: 1/8 resolution     (MaxPool2d(8))
Scale 4: 1/16 resolution    (MaxPool2d(16))
```

Each scale extracts 128-channel features through:
- Conv2d(1→64) + ReLU
- Conv2d(64→128) + ReLU
- MaxPool2d (for scales 1-4)

#### 3. FSCN Adaptive Concatenation Module
- **Input**: Decoder feature (scale 0) + 4 encoder features (scales 1-4)
- **Features**:
  - Spatial attention for each encoder feature
  - Adaptive weighting of encoder features (learned)
  - CBAM-style channel and spatial attention
  - Edge-preserving convolutions with Sobel filters
  - Dilated convolutions for larger receptive field

#### 4. Progressive Refinement (3 Stages)
```python
Stage 1: 128 → 32 channels  (with channel attention)
Stage 2: 32 → 8 channels    (with upsampling 2x)
Stage 3: 8 → 1 channel      (final depth output)
```

#### 5. Edge Enhancement Module
- **Input**: Refined depth map
- **Output**: Edge attention weights (sigmoid)
- **Purpose**: Learn edge-aware blending weights

#### 6. Edge-Aware Fusion
```python
final_output = edge_weights * refined_depth + (1 - edge_weights) * original_depth
```

### Data Flow

```
RGB Input (B, 3, H, W)
    ↓
DepthPro Backbone (Frozen)
    ↓
Initial Depth (B, 1, H, W)
    ↓
Multi-Scale Feature Extraction (5 scales)
    ↓ [Scale 0] [Scale 1] [Scale 2] [Scale 3] [Scale 4]
    ↓     ↓         ↓         ↓         ↓
FSCN Adaptive Concatenation Module
    ↓
Progressive Refinement (3 stages)
    ↓
Edge Enhancement
    ↓
Edge-Aware Fusion
    ↓
Final Depth Output (B, 1, H, W)
```

### Key Improvements Over Standard DepthPro

1. **Multi-Scale Processing**: 5 different scales for comprehensive feature extraction
2. **Attention Mechanisms**: Channel and spatial attention (CBAM-style)
3. **Adaptive Fusion**: Learned weighting of features from different scales
4. **Edge Preservation**: Sobel filter-based edge enhancement
5. **Progressive Refinement**: Gradual upsampling with attention at each stage
6. **Edge-Aware Blending**: Learned combination of original and refined depth

### Parameter Comparison

- **Original DepthPro**: ~4K trainable parameters (simple MLP)
- **DepthPro + FSCN**: ~500K+ trainable parameters (comprehensive FSCN decoder)
- **Increase**: ~125x more parameters for significantly better fusion

### Training Strategy

- **Backbone**: Always frozen (inference-only)
- **FSCN Components**: Trainable in training mode
- **Fallback**: OOM-safe single-image processing
- **Compatibility**: Drop-in replacement for original DepthProInference

### Expected Benefits

1. **Better Detail Preservation**: Multi-scale fusion preserves fine details
2. **Improved Edge Quality**: Edge-preserving convolutions and enhancement
3. **Robust Feature Integration**: Attention-based adaptive fusion
4. **Enhanced Spatial Consistency**: CBAM-style spatial attention
5. **Flexible Architecture**: Handles various input sizes and batch dimensions 
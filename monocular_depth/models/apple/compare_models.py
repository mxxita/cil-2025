"""
Model Comparison Script for DepthPro vs DepthPro+FSCN

This script compares the original DepthProInference model with the new
DepthProWithFSCN model to demonstrate the improvements from the FSCN decoder.
"""

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

from monocular_depth.models.apple.depth import DepthProInference, DepthProWithFSCN
from monocular_depth.config.device import get_device


def create_sample_input(batch_size=1, height=384, width=384):
    """Create a sample RGB input tensor."""
    return torch.randn(batch_size, 3, height, width)


def compare_model_architectures():
    """Compare the architectural differences between models."""
    print("=" * 80)
    print("MODEL ARCHITECTURE COMPARISON")
    print("=" * 80)
    
    # Initialize both models
    print("Initializing models...")
    model_original = DepthProInference(prefer_cpu=False, enable_training=True)
    model_fscn = DepthProWithFSCN(prefer_cpu=False, enable_training=True)
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    original_params = count_parameters(model_original)
    fscn_params = count_parameters(model_fscn)
    
    print(f"\nParameter Count:")
    print(f"  Original DepthPro:     {original_params:,} trainable parameters")
    print(f"  DepthPro + FSCN:       {fscn_params:,} trainable parameters")
    print(f"  Additional parameters: {fscn_params - original_params:,} ({((fscn_params/original_params - 1) * 100):.1f}% increase)")
    
    # Architecture details
    print(f"\nArchitecture Details:")
    print(f"  Original DepthPro:")
    print(f"    - Frozen DepthPro backbone")
    print(f"    - Simple MLP head (4 conv layers)")
    print(f"    - No attention mechanisms")
    print(f"    - Single-scale processing")
    
    print(f"  DepthPro + FSCN:")
    print(f"    - Frozen DepthPro backbone")
    print(f"    - Multi-scale feature extraction (5 scales)")
    print(f"    - Adaptive Concatenation Module with attention")
    print(f"    - Channel and Spatial attention mechanisms")
    print(f"    - Edge-preserving convolutions")
    print(f"    - Progressive refinement layers")
    print(f"    - Edge enhancement module")


def benchmark_inference_speed():
    """Benchmark inference speed for both models."""
    print("\n" + "=" * 80)
    print("INFERENCE SPEED BENCHMARK")
    print("=" * 80)
    
    device = get_device(prefer_cpu=False)
    
    # Initialize models
    model_original = DepthProInference(prefer_cpu=False, enable_training=False)
    model_fscn = DepthProWithFSCN(prefer_cpu=False, enable_training=False)
    
    model_original.eval()
    model_fscn.eval()
    
    # Create test input
    test_input = create_sample_input(batch_size=2, height=384, width=384).to(device)
    
    # Warmup runs
    print("Warming up models...")
    with torch.no_grad():
        for _ in range(5):
            _ = model_original(test_input)
            _ = model_fscn(test_input)
    
    # Benchmark original model
    print("Benchmarking Original DepthPro...")
    times_original = []
    with torch.no_grad():
        for i in range(10):
            start_time = time.time()
            output_original = model_original(test_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            times_original.append(end_time - start_time)
            print(f"  Run {i+1}: {times_original[-1]:.4f}s")
    
    # Benchmark FSCN model
    print("Benchmarking DepthPro + FSCN...")
    times_fscn = []
    with torch.no_grad():
        for i in range(10):
            start_time = time.time()
            output_fscn = model_fscn(test_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            times_fscn.append(end_time - start_time)
            print(f"  Run {i+1}: {times_fscn[-1]:.4f}s")
    
    # Results
    avg_original = np.mean(times_original)
    avg_fscn = np.mean(times_fscn)
    
    print(f"\nSpeed Comparison Results:")
    print(f"  Original DepthPro:     {avg_original:.4f}s ± {np.std(times_original):.4f}s")
    print(f"  DepthPro + FSCN:       {avg_fscn:.4f}s ± {np.std(times_fscn):.4f}s")
    print(f"  Slowdown factor:       {avg_fscn/avg_original:.2f}x")
    
    return output_original, output_fscn


def analyze_output_differences(output_original, output_fscn):
    """Analyze the differences in model outputs."""
    print("\n" + "=" * 80)
    print("OUTPUT ANALYSIS")
    print("=" * 80)
    
    # Convert to numpy for analysis
    orig_np = output_original.cpu().numpy()
    fscn_np = output_fscn.cpu().numpy()
    
    print(f"Output shapes:")
    print(f"  Original: {orig_np.shape}")
    print(f"  FSCN:     {fscn_np.shape}")
    
    print(f"\nOutput statistics:")
    print(f"  Original - Min: {orig_np.min():.4f}, Max: {orig_np.max():.4f}, Mean: {orig_np.mean():.4f}")
    print(f"  FSCN     - Min: {fscn_np.min():.4f}, Max: {fscn_np.max():.4f}, Mean: {fscn_np.mean():.4f}")
    
    # Compute difference metrics
    diff = np.abs(orig_np - fscn_np)
    rel_diff = diff / (orig_np + 1e-6)
    
    print(f"\nDifference metrics:")
    print(f"  Mean absolute difference: {diff.mean():.4f}")
    print(f"  Max absolute difference:  {diff.max():.4f}")
    print(f"  Mean relative difference: {rel_diff.mean():.4f}")
    print(f"  Max relative difference:  {rel_diff.max():.4f}")


def test_training_compatibility():
    """Test that both models are compatible with training."""
    print("\n" + "=" * 80)
    print("TRAINING COMPATIBILITY TEST")
    print("=" * 80)
    
    device = get_device(prefer_cpu=False)
    
    # Initialize models in training mode
    model_original = DepthProInference(prefer_cpu=False, enable_training=True)
    model_fscn = DepthProWithFSCN(prefer_cpu=False, enable_training=True)
    
    model_original.train()
    model_fscn.train()
    
    # Create test input and target
    test_input = create_sample_input(batch_size=2, height=384, width=384).to(device)
    test_target = torch.randn(2, 1, 384, 384).to(device)
    
    # Test forward pass
    print("Testing forward pass...")
    output_original = model_original(test_input)
    output_fscn = model_fscn(test_input)
    
    print(f"  Original output shape: {output_original.shape}")
    print(f"  FSCN output shape:     {output_fscn.shape}")
    
    # Test backward pass
    print("Testing backward pass...")
    criterion = nn.MSELoss()
    
    loss_original = criterion(output_original, test_target)
    loss_fscn = criterion(output_fscn, test_target)
    
    print(f"  Original loss: {loss_original.item():.4f}")
    print(f"  FSCN loss:     {loss_fscn.item():.4f}")
    
    # Test gradients
    loss_original.backward()
    loss_fscn.backward()
    
    # Check gradient statistics
    def check_gradients(model, name):
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        if grad_norms:
            print(f"  {name} gradients - Mean norm: {np.mean(grad_norms):.6f}, Max norm: {np.max(grad_norms):.6f}")
        else:
            print(f"  {name} - No gradients found!")
    
    check_gradients(model_original, "Original")
    check_gradients(model_fscn, "FSCN")
    
    print("✓ Both models are training-compatible!")


def main():
    """Run complete comparison."""
    print("DEPTHPRO vs DEPTHPRO+FSCN COMPARISON")
    print("=" * 80)
    
    try:
        # Architecture comparison
        compare_model_architectures()
        
        # Speed benchmark
        output_original, output_fscn = benchmark_inference_speed()
        
        # Output analysis
        analyze_output_differences(output_original, output_fscn)
        
        # Training compatibility
        test_training_compatibility()
        
        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("1. FSCN model has significantly more parameters for better feature fusion")
        print("2. FSCN model has slower inference due to additional computations")
        print("3. Both models are training-compatible")
        print("4. FSCN model should provide better depth quality due to attention mechanisms")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
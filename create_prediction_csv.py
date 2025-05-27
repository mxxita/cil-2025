#!/usr/bin/env python3
"""
Create Kaggle submission CSV from trained depth estimation model.
This script loads the best checkpoint and generates predictions for the test set.
"""

import argparse
import os
import sys
import time
import base64
import zlib
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from monocular_depth.models.apple.depth import DepthProInference
from monocular_depth.data.dataset import DepthDataset
from monocular_depth.data.transforms import test_transform
from monocular_depth.utils.helpers import custom_collate_fn
from monocular_depth.config.device import get_device


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load model from checkpoint file."""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Create model
    model = DepthProInference(
        prefer_cpu=(device.type == 'cpu'),
        enable_training=True  # Need this for the MLP head
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # New format from cluster training (with metadata)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint and checkpoint['metrics']:
            print("Checkpoint metrics:")
            for metric, value in checkpoint['metrics'].items():
                print(f"  {metric}: {value:.6f}")
    else:
        # Old format (direct state_dict)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully (direct state_dict format)")
    
    model = model.to(device)
    model.eval()
    
    return model


def create_test_dataset(test_dir: str, test_list: str, input_size: tuple) -> DepthDataset:
    """Create test dataset."""
    # Check if test list exists, if not create it
    if not os.path.exists(test_list):
        print(f"Test list not found at {test_list}, creating from directory...")
        test_files = []
        
        # Find all RGB images in test directory
        for file in os.listdir(test_dir):
            if file.endswith('_rgb.png'):
                test_files.append(file)
        
        # Sort for consistent ordering
        test_files.sort()
        
        # Write test list (format: just the RGB filename for test set)
        with open(test_list, 'w') as f:
            for file in test_files:
                f.write(f"{file}\n")
        
        print(f"Created test list with {len(test_files)} files")
    
    # Create dataset
    dataset = DepthDataset(
        data_dir=test_dir,
        list_file=test_list,
        input_size=input_size,
        transform=test_transform,
        has_gt=False  # Test set has no ground truth
    )
    
    return dataset


def generate_predictions(model: nn.Module, test_loader: DataLoader, device: torch.device) -> dict:
    """Generate predictions for test set."""
    model.eval()
    predictions = {}
    
    print("Generating predictions...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing")):
            if batch is None:
                continue
            
            # For test set, batch is (inputs, filenames) not (inputs, targets, filenames)
            if len(batch) == 2:
                inputs, filenames = batch
            else:
                inputs, _, filenames = batch
            
            inputs = inputs.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            
            # Ensure outputs are on CPU
            outputs = outputs.cpu().numpy()
            
            # Store predictions
            for i, filename in enumerate(filenames):
                # Remove file extension and _rgb suffix to get sample ID
                sample_id = filename.replace('_rgb.png', '').replace('.png', '')
                
                # Get depth prediction (squeeze to remove batch and channel dims if present)
                depth_pred = outputs[i]
                if depth_pred.ndim > 2:
                    depth_pred = depth_pred.squeeze()
                
                predictions[sample_id] = depth_pred
                
                if batch_idx == 0 and i == 0:  # Debug first prediction
                    print(f"Sample prediction shape: {depth_pred.shape}")
                    print(f"Sample prediction range: [{depth_pred.min():.4f}, {depth_pred.max():.4f}]")
    
    print(f"Generated predictions for {len(predictions)} samples")
    return predictions


def compress_depth_values(depth_values):
    """Compress depth values using zlib and base64 encoding."""
    # Convert depth values to bytes (rounded to 2 decimal places)
    depth_bytes = ",".join(f"{x:.2f}" for x in depth_values).encode("utf-8")
    # Compress using zlib
    compressed = zlib.compress(depth_bytes, level=9)  # level 9 is maximum compression
    # Encode as base64 for safe CSV storage
    return base64.b64encode(compressed).decode("utf-8")


def create_kaggle_csv(predictions: dict, output_file: str, format_type: str = "compressed"):
    """
    Create Kaggle submission CSV file matching the existing format.
    
    Args:
        predictions: Dictionary of {sample_id: depth_array}
        output_file: Path to output CSV file
        format_type: "compressed" (default), "flattened", or "files"
    """
    print(f"Creating Kaggle submission CSV: {output_file}")
    
    if format_type == "compressed":
        # Format matching existing script: compressed base64 encoding
        # CSV format: id, Depths (where Depths is compressed)
        
        ids = []
        depths_list = []
        
        for sample_id, depth_array in tqdm(predictions.items(), desc="Compressing depth maps"):
            # Flatten the depth array and round to 2 decimal places
            flattened_depth = np.round(depth_array.flatten(), 2)
            
            # Compress the depth values
            compressed_depths = compress_depth_values(flattened_depth)
            
            ids.append(sample_id)
            depths_list.append(compressed_depths)
        
        # Create DataFrame with exact same column names as existing script
        df = pd.DataFrame({
            "id": ids,
            "Depths": depths_list,
        })
        
        df.to_csv(output_file, index=False)
        
    elif format_type == "flattened":
        # Format 1: Flattened depth values
        # CSV format: sample_id, pixel_0, pixel_1, ..., pixel_N
        
        rows = []
        for sample_id, depth_array in tqdm(predictions.items(), desc="Formatting"):
            # Flatten the depth array
            flattened = depth_array.flatten()
            
            # Create row: [sample_id, pixel_0, pixel_1, ...]
            row = [sample_id] + flattened.tolist()
            rows.append(row)
        
        # Create column names
        sample_shape = next(iter(predictions.values())).shape
        total_pixels = sample_shape[0] * sample_shape[1]
        columns = ['sample_id'] + [f'pixel_{i}' for i in range(total_pixels)]
        
        # Create DataFrame and save
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(output_file, index=False)
        
    else:
        # Format 3: Simple format - just save as separate files and list them
        # CSV format: sample_id, depth_file_path
        
        output_dir = Path(output_file).parent / "depth_predictions"
        output_dir.mkdir(exist_ok=True)
        
        rows = []
        for sample_id, depth_array in tqdm(predictions.items(), desc="Saving depth files"):
            # Save depth array as .npy file
            depth_file = output_dir / f"{sample_id}_depth.npy"
            np.save(depth_file, depth_array)
            
            # Add to CSV
            rows.append([sample_id, str(depth_file)])
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=['sample_id', 'depth_file'])
        df.to_csv(output_file, index=False)
    
    print(f"Saved {len(predictions)} predictions to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate Kaggle submission from trained model')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--test-dir', default='monocular_depth/data/test', help='Test data directory')
    parser.add_argument('--test-list', default='monocular_depth/data/test_list.txt', help='Test list file')
    parser.add_argument('--output', default='kaggle_submission.csv', help='Output CSV file')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--input-size', nargs=2, type=int, default=[426, 560], help='Input image size (H W)')
    parser.add_argument('--format', choices=['compressed', 'flattened', 'files'], default='compressed', 
                        help='Output format for Kaggle submission (compressed matches existing format)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    print("=== Kaggle Submission Generator ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test directory: {args.test_dir}")
    print(f"Output file: {args.output}")
    print(f"Format: {args.format}")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    
    # Get device
    if args.device == 'auto':
        device = get_device(prefer_cpu=False)  # Prefer GPU if available
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, device)
    
    # Create test dataset
    test_dataset = create_test_dataset(
        args.test_dir, 
        args.test_list, 
        tuple(args.input_size)
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        collate_fn=custom_collate_fn
    )
    
    # Generate predictions
    start_time = time.time()
    predictions = generate_predictions(model, test_loader, device)
    inference_time = time.time() - start_time
    
    print(f"Inference completed in {inference_time:.2f} seconds")
    print(f"Average time per sample: {inference_time/len(predictions):.3f} seconds")
    
    # Create Kaggle submission CSV
    create_kaggle_csv(predictions, args.output, args.format)
    
    print(f"\n✅ Kaggle submission created: {args.output}")
    print("\nSubmission summary:")
    print(f"  - Total samples: {len(predictions)}")
    print(f"  - Format: {args.format}")
    print(f"  - File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")
    
    # Show first few entries
    df = pd.read_csv(args.output)
    print(f"\nFirst 5 entries:")
    print(df.head())
    
    return 0


if __name__ == '__main__':
    exit(main()) 
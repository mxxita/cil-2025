"""Test script to verify data loading works correctly."""

import unittest
import torch
import os
from torch.utils.data import DataLoader
from monocular_depth.data.dataset import DepthDataset
from monocular_depth.data.transforms import train_transform, test_transform
from monocular_depth.config.paths import train_dir, test_dir, train_list_file, test_list_file
from monocular_depth.utils.helpers import custom_collate_fn
from monocular_depth.models.apple.config import INPUT_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY

class TestDataLoading(unittest.TestCase):
    """Test case for verifying data loading."""

    def setUp(self):
        """Set up test fixtures."""
        print("\nDebug information:")
        print(f"Train directory: {train_dir}")
        print(f"Train list file: {train_list_file}")
        print(f"Train list exists: {os.path.exists(train_list_file)}")
        
        # Print first few lines of train list
        if os.path.exists(train_list_file):
            print("\nFirst few lines of train_list.txt:")
            with open(train_list_file, 'r') as f:
                for i, line in enumerate(f):
                    if i < 3:  # Print first 3 lines
                        print(f"  {line.strip()}")
                        # Check if files exist
                        rgb_file, depth_file = line.strip().split()
                        rgb_path = os.path.join(train_dir, rgb_file)
                        depth_path = os.path.join(train_dir, depth_file)
                        print(f"  RGB exists: {os.path.exists(rgb_path)}")
                        print(f"  Depth exists: {os.path.exists(depth_path)}")
        
        # Create training dataset
        self.train_dataset = DepthDataset(
            data_dir=train_dir,
            list_file=train_list_file,
            input_size=INPUT_SIZE,
            transform=train_transform,
            has_gt=True
        )
        
        # Create test dataset
        self.test_dataset = DepthDataset(
            data_dir=test_dir,
            list_file=test_list_file,
            input_size=INPUT_SIZE,
            transform=test_transform,
            has_gt=False
        )

    def test_dataset_creation(self):
        """Test that datasets are created successfully."""
        self.assertIsNotNone(self.train_dataset)
        self.assertIsNotNone(self.test_dataset)
        print(f"\nTraining dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")
        print(f"Skipped training files: {self.train_dataset.get_skipped_count()}")
        print(f"Skipped test files: {self.test_dataset.get_skipped_count()}")

    def test_training_sample(self):
        """Test loading a single training sample."""
        # Try loading first sample
        print("\nTrying to load first training sample...")
        if len(self.train_dataset.file_pairs) > 0:
            rgb_file, depth_file = self.train_dataset.file_pairs[0]
            print(f"First pair - RGB: {rgb_file}, Depth: {depth_file}")
            print(f"Full paths:")
            print(f"  RGB: {os.path.join(train_dir, rgb_file)}")
            print(f"  Depth: {os.path.join(train_dir, depth_file)}")
        
        sample = self.train_dataset[0]
        self.assertIsNotNone(sample, "Training sample should not be None")
        
        if sample is not None:
            rgb, depth, filename = sample
            print("\nTraining sample shapes:")
            print(f"RGB shape: {rgb.shape}")
            print(f"Depth shape: {depth.shape}")
            print(f"Filename: {filename}")
            
            # Check shapes
            self.assertEqual(rgb.shape[0], 3, "RGB should have 3 channels")
            self.assertEqual(depth.shape[0], 1, "Depth should have 1 channel")
            self.assertEqual(rgb.shape[1:], INPUT_SIZE, "RGB should match input size")
            self.assertEqual(depth.shape[1:], INPUT_SIZE, "Depth should match input size")

    def test_data_loader(self):
        """Test creating and using a data loader."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            collate_fn=custom_collate_fn
        )
        
        # Try loading a batch
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                print(f"\nWarning: Batch {batch_idx} is None")
                continue
                
            inputs, targets, filenames = batch
            print(f"\nBatch {batch_idx} shapes:")
            print(f"Inputs shape: {inputs.shape}")
            print(f"Targets shape: {targets.shape}")
            print(f"Number of filenames: {len(filenames)}")
            
            # Check shapes
            self.assertEqual(inputs.shape[0], BATCH_SIZE, "Batch size should match")
            self.assertEqual(inputs.shape[1], 3, "Input should have 3 channels")
            self.assertEqual(targets.shape[1], 1, "Target should have 1 channel")
            self.assertEqual(inputs.shape[2:], INPUT_SIZE, "Input should match input size")
            self.assertEqual(targets.shape[2:], INPUT_SIZE, "Target should match input size")
            
            # Only test first batch
            break

if __name__ == '__main__':
    unittest.main(verbosity=2) 
"""Test script to verify all imports work correctly."""

import unittest
import importlib
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class TestImports(unittest.TestCase):
    """Test case for verifying all imports work correctly."""

    def test_data_imports(self):
        """Test imports from data module."""
        from monocular_depth.data import DepthDataset, train_transform, test_transform
        
        # Verify imports exist
        self.assertIsNotNone(DepthDataset)
        self.assertIsNotNone(train_transform)
        self.assertIsNotNone(test_transform)

    def test_training_imports(self):
        """Test imports from training module."""
        from monocular_depth.training import train_model, SILogLoss
        
        # Verify imports exist
        self.assertIsNotNone(train_model)
        self.assertIsNotNone(SILogLoss)

    def test_models_imports(self):
        """Test imports from models module."""
        from monocular_depth.models import DepthProInference
        from monocular_depth.models.apple import DepthProInference as AppleDepthPro
        
        # Verify imports exist
        self.assertIsNotNone(DepthProInference)
        self.assertIsNotNone(AppleDepthPro)
        self.assertEqual(DepthProInference, AppleDepthPro)

    def test_utils_imports(self):
        """Test imports from utils module."""
        from monocular_depth.utils import (
            ensure_dir,
            target_transform,
            custom_collate_fn,
            print_tqdm
        )
        
        # Verify imports exist
        self.assertIsNotNone(ensure_dir)
        self.assertIsNotNone(target_transform)
        self.assertIsNotNone(custom_collate_fn)
        self.assertIsNotNone(print_tqdm)

    def test_module_imports(self):
        """Test importing entire modules."""
        modules = [
            'monocular_depth',
            'monocular_depth.data',
            'monocular_depth.training',
            'monocular_depth.models',
            'monocular_depth.models.apple',
            'monocular_depth.utils'
        ]
        
        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)

if __name__ == '__main__':
    unittest.main() 
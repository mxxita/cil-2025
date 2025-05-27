import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from monocular_depth.models.apple.config import DEVICE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, INPUT_SIZE
from monocular_depth.data.dataset import DepthDataset
from monocular_depth.models.depth_pro import DepthProInference  # Use the simpler wrapper
from monocular_depth.data.transforms import test_transform
from monocular_depth.config.paths import train_dir, test_dir, train_list_file, test_list_file, results_dir, predictions_dir
from monocular_depth.utils.helpers import (
    ensure_dir,
    custom_collate_fn,
)

from monocular_depth.inference.evaluate import evaluate_model, generate_test_predictions


def main():
    print("=== ONE-SHOT EVALUATION WITH DEPTHPRO BACKBONE (SIMPLE WRAPPER) ===")
    
    # Create datasets
    print("Creating datasets...")
    train_full_dataset = DepthDataset(
        data_dir=train_dir,
        list_file=train_list_file,
        input_size=INPUT_SIZE,
        transform=test_transform,  # Use test transform for consistent evaluation
        has_gt=True,
    )
    print(
        f"Skipped files in training dataset: {train_full_dataset.get_skipped_count()}"
    )
    test_dataset = DepthDataset(
        data_dir=test_dir,
        list_file=test_list_file,
        input_size=INPUT_SIZE,
        transform=test_transform,
        has_gt=False,
    )
    print(f"Skipped files in test dataset: {test_dataset.get_skipped_count()}")

    # Split training dataset to get validation set
    total_size = len(train_full_dataset)
    train_size = int(0.85 * total_size)
    val_size = total_size - train_size
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full_dataset, [train_size, val_size]
    )

    # Create data loaders (only need validation and test for evaluation)
    print("Creating data loaders...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=custom_collate_fn,
    )
    print(
        f"Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}"
    )

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print(f"Initially allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    # Load pretrained model - Simple DepthPro wrapper
    print("Loading DepthPro model with simple wrapper...")
    
    model = DepthProInference()  # Simple wrapper class
    model.to(DEVICE)
    model.eval()  # Set to evaluation mode

    print(f"Using device: {DEVICE}")
    print("Model configuration: DepthPro with simple wrapper (inference only)")
    if torch.cuda.is_available():
        print(
            f"Memory allocated after model init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
        )

    # Evaluate model on validation set
    print("\n=== EVALUATING DEPTHPRO WITH SIMPLE WRAPPER ON VALIDATION SET ===")
    with torch.no_grad():
        metrics = evaluate_model(model, val_loader, DEVICE, results_dir)

    print("\nValidation Metrics (DepthPro Simple Wrapper):")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # Save validation metrics
    with open(os.path.join(results_dir, "depthpro_simple_validation_metrics.txt"), "w") as f:
        f.write("=== DepthPro Simple Wrapper - Validation Metrics ===\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

    # Generate test predictions
    print("\n=== GENERATING PREDICTIONS FOR TEST SET ===")
    with torch.no_grad():
        generate_test_predictions(model, test_loader, DEVICE, predictions_dir)

    print(f"\nResults saved to {results_dir}")
    print(f"All test depth map predictions saved to {predictions_dir}")
    print("=== ONE-SHOT EVALUATION COMPLETE ===")


if __name__ == "__main__":
    main()
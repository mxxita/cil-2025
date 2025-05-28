import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from monocular_depth.models.apple.config import DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, INPUT_SIZE
from monocular_depth.data.dataset import DepthDataset
from monocular_depth.models.apple.depth import DepthProWithFSCN
from monocular_depth.data.transforms import train_transform, test_transform
from monocular_depth.config.paths import train_dir, test_dir, train_list_file, test_list_file, results_dir, predictions_dir
from monocular_depth.training.loss import SILogLoss
from monocular_depth.models.apple.encoder_new import CustomFeatureFusionLayer, CustomFSCNFusionStage
from monocular_depth.training.train import train_model # Train model
from monocular_depth.utils.helpers import (
    ensure_dir,
    custom_collate_fn,
)

from monocular_depth.inference.evaluate import evaluate_model, generate_test_predictions


def main():
    # Create datasets
    print("Creating datasets...")
    train_full_dataset = DepthDataset(
        data_dir=train_dir,
        list_file=train_list_file,
        input_size=INPUT_SIZE,
        transform=train_transform,
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

    # Split training dataset
    total_size = len(train_full_dataset)
    train_size = int(0.85 * total_size)
    val_size = total_size - train_size
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full_dataset, [train_size, val_size]
    )

        # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=False,
        collate_fn=custom_collate_fn,
    )
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
        f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}"
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

    # Load pretrained model
    print("Loading pretrained model...")

    # Initialize DepthPro model with trainable MLP head
    model = DepthProWithFSCN(prefer_cpu=False, enable_training=True)

    model.to(DEVICE)

    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(
            f"Memory allocated after model init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
        )

    # Define loss and optimizer
    criterion = SILogLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Finetune model
    print("Starting finetuning...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        NUM_EPOCHS,
        DEVICE,
        results_dir,
        in_epoch_validation=True,
    )

    # Evaluate model
    print("Evaluating model on validation set...")
    metrics = evaluate_model(model, val_loader, DEVICE, results_dir)


    print("\nValidation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    with open(os.path.join(results_dir, "validation_metrics.txt"), "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

    # Generate test predictions
    print("Generating predictions for test set...")
    generate_test_predictions(model, test_loader, DEVICE, predictions_dir)

    print(f"Results saved to {results_dir}")
    print(f"All test depth map predictions saved to {predictions_dir}")


if __name__ == "__main__":
    main()
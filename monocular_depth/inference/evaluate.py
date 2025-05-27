import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from monocular_depth.utils.helpers import ensure_dir


def evaluate_model(model, val_loader, device, results_dir):
    """Evaluate the model and compute metrics on validation set"""
    model.eval()

    mae = 0.0
    rmse = 0.0
    rel = 0.0
    delta1 = 0.0
    delta2 = 0.0
    delta3 = 0.0
    sirmse = 0.0

    total_samples = 0
    target_shape = None

    with torch.no_grad():
        for inputs, targets, filenames in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            if target_shape is None:
                target_shape = targets.shape

            # Forward pass
            outputs = model(inputs).predicted_depth

            # Resize outputs to match target dimensions
            if outputs.dim() == 3:
                outputs = outputs.unsqueeze(1)
            outputs = nn.functional.interpolate(
                outputs,
                size=targets.shape[-2:],  # Match height and width of targets
                mode="bilinear",
                align_corners=True,
            )

            # Calculate metrics
            abs_diff = torch.abs(outputs - targets)
            mae += torch.sum(abs_diff).item()
            rmse += torch.sum(torch.pow(abs_diff, 2)).item()
            rel += torch.sum(abs_diff / (targets + 1e-6)).item()

            # Calculate scale-invariant RMSE for each image in the batch
            for i in range(batch_size):
                # Convert tensors to numpy arrays
                pred_np = outputs[i].cpu().squeeze().numpy()
                target_np = targets[i].cpu().squeeze().numpy()

                EPSILON = 1e-6

                # mask for valid pixels
                valid_mask = target_np > EPSILON
                if not np.any(valid_mask):
                    continue

                target_valid = target_np[valid_mask]
                pred_valid = pred_np[valid_mask]

                pred_valid = pred_valid.flatten()

                log_target = np.log(target_valid)
                pred_valid = np.where(pred_valid > EPSILON, pred_valid, EPSILON)
                log_pred = np.log(pred_valid)

                diff = log_pred - log_target
                diff_mean = np.mean(diff)

                # Calculate RMSE for this image
                sirmse += np.sqrt(np.mean((diff - diff_mean) ** 2))

            # Calculate thresholded accuracy
            max_ratio = torch.max(
                outputs / (targets + 1e-6), targets / (outputs + 1e-6)
            )
            delta1 += torch.sum(max_ratio < 1.25).item()
            delta2 += torch.sum(max_ratio < 1.25**2).item()
            delta3 += torch.sum(max_ratio < 1.25**3).item()

            # Save some sample predictions
            if total_samples <= 5 * batch_size:
                for i in range(min(batch_size, 5)):
                    idx = total_samples - batch_size + i

                    # Convert tensors to numpy arrays
                    input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
                    target_np = targets[i].cpu().squeeze().numpy()
                    output_np = outputs[i].cpu().numpy().squeeze(0)

                    # Normalize for visualization
                    input_np = (input_np - input_np.min()) / (
                        input_np.max() - input_np.min() + 1e-6
                    )

                    # Create visualization
                    plt.figure(figsize=(15, 5))

                    plt.subplot(1, 3, 1)
                    plt.imshow(input_np)
                    plt.title("RGB Input")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(target_np, cmap="plasma")
                    plt.title("Ground Truth Depth")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(output_np, cmap="plasma")
                    plt.title("Predicted Depth")
                    plt.axis("off")

                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, f"sample_{idx}.png"))
                    plt.close()

            # Free up memory
            del inputs, targets, outputs, abs_diff, max_ratio

        # Clear CUDA cache
        torch.cuda.empty_cache()

    # Calculate final metrics using stored target shape
    total_pixels = (
        target_shape[1] * target_shape[2] * target_shape[3]
    )  # channels * height * width
    mae /= total_samples * total_pixels
    rmse = np.sqrt(rmse / (total_samples * total_pixels))
    rel /= total_samples * total_pixels
    sirmse = sirmse / total_samples
    delta1 /= total_samples * total_pixels
    delta2 /= total_samples * total_pixels
    delta3 /= total_samples * total_pixels

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "siRMSE": sirmse,
        "REL": rel,
        "Delta1": delta1,
        "Delta2": delta2,
        "Delta3": delta3,
    }

    return metrics


def generate_test_predictions(model, test_loader, device, predictions_dir):
    """Generate predictions for the test set without ground truth"""
    """DONT CHANGE"""

    model.eval()

    # Ensure predictions directory exists
    ensure_dir(predictions_dir)

    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Generating Test Predictions"):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # Forward pass
            outputs = model(inputs).predicted_depth

            # Resize outputs to match original input dimensions (426x560)
            if outputs.dim() == 3:
                outputs = outputs.unsqueeze(1)
            outputs = nn.functional.interpolate(
                outputs,
                size=(426, 560),  # Original input dimensions
                mode="bilinear",
                align_corners=True,
            )

            # Save all test predictions
            for i in range(batch_size):
                # Get filename without extension
                filename = filenames[i].split(" ")[1]

                # Save depth map prediction as numpy array
                depth_pred = outputs[i].cpu().squeeze().numpy()
                np.save(os.path.join(predictions_dir, f"{filename}"), depth_pred)

            # Clean up memory
            del inputs, outputs

        # Clear cache after test predictions
        torch.cuda.empty_cache()
# Monocular Depth Estimation

This project implements a modular deep learning solution for monocular depth estimation using PyTorch. The implementation uses a U-Net architecture to predict depth maps from single RGB images.

## Project Structure

```
monocular_depth/
├── config/             # Configuration files
├── monocular_depth/    # Main package
│   ├── data/          # Dataset and data loading
│   ├── models/        # Model architectures
│   ├── training/      # Training and evaluation
│   └── utils/         # Helper functions
└── tests/             # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd monocular_depth
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in the following structure:
```
data/
├── train/
│   ├── train/         # Training RGB images
│   └── train_list.txt # List of training image pairs
└── test/
    ├── test/          # Test RGB images
    └── test_list.txt  # List of test images
```

2. Configure your training parameters in `config/config.yaml`

3. Run training:
```bash
python -m monocular_depth.main
```

## Features

- Modular U-Net architecture for depth estimation
- Custom dataset loader for RGB-depth pairs
- Training with validation and early stopping
- Comprehensive evaluation metrics
- Test prediction generation
- GPU support with memory optimization

## Metrics

The model is evaluated using the following metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- siRMSE (Scale-invariant RMSE)
- REL (Relative Error)
- Delta1, Delta2, Delta3 (Thresholded accuracy)

## Path Configuration

The project uses environment variables to handle paths across different environments (euler cluster, local machine, etc.).

### Setting Up Paths

1. Copy the environment template:
   ```bash
   cp .env.template .env
   ```

2. Edit `.env` to set your paths:
   ```bash
   # For euler cluster
   CIL_BASE_PATH=/cluster/home/yourusername/cil-2025
   CIL_DATA_PATH=/cluster/scratch/yourusername/courses/cil/monocular_depth
   CIL_MODELS_PATH=/cluster/home/yourusername/cil-2025/ml-depth-pro/models
   CIL_OUTPUTS_PATH=/cluster/home/yourusername/cil-2025/ml-depth-pro/outputs
   CIL_CHECKPOINTS_PATH=/cluster/home/yourusername/cil-2025/checkpoints

   # For local machine
   CIL_BASE_PATH=/path/to/your/project
   CIL_DATA_PATH=/path/to/your/data
   CIL_MODELS_PATH=/path/to/your/models
   CIL_OUTPUTS_PATH=/path/to/your/outputs
   CIL_CHECKPOINTS_PATH=/path/to/your/checkpoints
   ```

3. The `.env` file is git-ignored, so your paths won't be committed.

### Using Paths in Code

```python
from monocular_depth.config.paths import (
    get_test_image_path,
    get_output_path,
    get_checkpoint_path
)

# Get paths
test_image = get_test_image_path()
output_dir = get_output_path('predictions')
checkpoint = get_checkpoint_path('model.pt')
```

### Default Paths

If no environment variables are set, the code will use default paths based on the detected environment (euler or local).

## License

[Your chosen license] 
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

## License

[Your chosen license] 
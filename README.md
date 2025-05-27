# Monocular Depth Estimation

A modular Python package for monocular depth estimation using deep learning models, including custom U-Net architecture and Apple's DepthPro model integration.

## Project Structure

```
monocular_depth/
├── data/                   # Data loading and preprocessing
│   ├── dataset.py         # Custom dataset classes
│   └── transforms.py      # Data augmentation and transforms
├── models/                # Model architectures
│   └── depth_pro.py       # DepthPro model wrapper
│   └── apple/              # Apple DepthPro model integration
│       └── depth.py        # DepthProInference wrapper and utilities
│       └── main.py         # Main entry point for DepthPro model usage
├── inference/             # Inference scripts and utilities
│   └── create_prediction_csv.py  # Convert predictions to CSV format
├── utils/                 # Utility functions
│   └── helpers.py         # General helper functions
└── config/
│   └── paths.py         # UPDATE
│   └── device.py         # General helper functions
└── __init__.py


ml-depth-pro/             # Apple's DepthPro model (git submodule)
requirements.txt          # Python dependencies
setup.py                  # Package installation
```


## 🛠️ Installation

### 1 Clone the repository with submodules

```bash
git clone --recurse-submodules https://github.com/mxxita/cil-2025.git
cd cil-2025
```

---

### 2 (Optional) If submodules weren’t pulled correctly

```bash
git submodule update --init --recursive
```

---

### 3 Create and activate the Conda environment

```bash
conda env create -f environment.yml
conda activate depth-pro
```

---

### 4 Install DepthPro (Apple submodule)

```bash
cd ml-depth-pro
pip install -e .
cd ..
```

---

### 5 Install your own project (monocular_depth)

```bash
pip install -e .
```
"""



## Usage

### Basic Inference with DepthPro

#### Using Command Line

```bash
# Run prediction on a single image:
depth-pro-run -i ./data/example.jpg
# Run `depth-pro-run -h` for available options.
```

#### Using Python API

```python
from PIL import Image
import depth_pro

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image. Habe f_px ignoriert
image, _, f_px = depth_pro.load_rgb("path/to/your/image.jpg")
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.
```

#### Using Project Wrapper

```python
from monocular_depth.models.depth_pro import DepthProInference
from PIL import Image

# Initialize model
depth_model = DepthProInference()

# Load and process image
image = Image.open("path/to/your/image.jpg")
depth_map = depth_model.predict(image)

# Visualize results
depth_model.visualize(image, depth_map, save_path="output.png")
```

### Custom Dataset Loading

```python
from monocular_depth.data.dataset import DepthDataset
from monocular_depth.data.transforms import get_transforms

# Create dataset
transforms = get_transforms(image_size=(480, 640))
dataset = DepthDataset(
    data_dir="path/to/data",
    list_file="train_list.txt",
    transforms=transforms
)

# Use with DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```
## Data Format

### Dataset Structure

```
data/ (wie am student cluster)
├── train_list.txt         # Training file pairs
├── val_list.txt           # Validation file pairs
└── test_list.txt          # Test file list
```
```

## Inference Pipeline

### Generate Predictions

```python
from monocular_depth.models.depth_pro import DepthProInference
import numpy as np
import os

# Initialize model
model = DepthProInference()

# Process test images
test_dir = "data/rgb"
output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

for img_file in os.listdir(test_dir):
    if img_file.endswith(('.jpg', '.png')):
        img_path = os.path.join(test_dir, img_file)
        depth_map = model.predict(img_path)
        
        # Save as numpy array
        output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}.npy")
        np.save(output_path, depth_map)
```

### Create Submission CSV

Use the provided script to convert predictions to CSV format:

```bash
cd monocular_depth/inference
python create_prediction_csv.py
```
Hier muss man nur die Pfade anpassen

## Model Information

### DepthPro Model

- **Source**: Apple's ML-DepthPro
- **Architecture**: Vision Transformer-based depth estimation
- **Input**: RGB images (any resolution)
- **Output**: Metric depth maps
- **Pretrained**: Yes (on diverse datasets)

### Key Features

- **Metric Depth**: Produces depth in real-world units (meters)
- **High Resolution**: Supports high-resolution input images
- **Fast Inference**: Optimized for real-time applications
- **Robust**: Works across diverse scenes and conditions

## Evaluation Metrics

The project includes standard depth estimation metrics:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **siRMSE**: Scale-Invariant RMSE
- **REL**: Mean Relative Error
- **Delta1, Delta2, Delta3**: Threshold accuracy metrics

## Dependencies

Key dependencies (see `requirements.txt` for complete list):

- `torch >= 1.9.0`
- `torchvision >= 0.10.0`
- `numpy`
- `Pillow`
- `matplotlib`
- `tqdm`
- `pyyaml`
- `pandas`

## Troubleshooting

## Citation

If you use this code, please cite the original DepthPro paper:

```bibtex
@article{depth_pro,
  title={Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
``` 
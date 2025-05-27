
# In paths.py
import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# PATHS
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))
DATA_DIR = os.getenv("DATA_DIR", "/cluster/scratch/mariberger/monocular_depth/data/")

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_dir = os.path.join(DATA_DIR, "train/")
test_dir = os.path.join(DATA_DIR, "test/")
train_list_file = os.path.join(DATA_DIR, "train_list.txt")
test_list_file = os.path.join(DATA_DIR, "test_list.txt")
output_dir = OUTPUT_DIR
results_dir = os.path.join(output_dir, "results")
predictions_dir = os.path.join(output_dir, "predictions")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)
import torch

BATCH_SIZE = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#INPUT_SIZE = (384, 384)
INPUT_SIZE = (426, 560)
NUM_WORKERS = 1
PIN_MEMORY = True
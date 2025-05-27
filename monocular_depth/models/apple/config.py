import torch

BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 1
DEVICE = torch.device('cpu') #if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = (426, 560)
NUM_WORKERS = 1
PIN_MEMORY = False


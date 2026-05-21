import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model
DROPOUT = 0.5

# data.py
BATCH_SIZE = 32
IMG_SIZE = 128

# train
LR = 0.001
EPOCHS = 10
PATIENCE = 4



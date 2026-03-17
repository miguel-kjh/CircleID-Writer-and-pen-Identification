import os

TASK = "writer"  # "writer" or "pen"

DATASET_DIR = "dataset/raw/"
IMAGE_DIR = "dataset/"
OUTPUT_DIR = "results/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
IMG_SIZE = 224
SEED = 0

# Holdout validation fraction (set 0.0 to disable)
VAL_FRAC = 0.2

# Writer task only: Below this confidence, writers are predicted as unknown (-1)
WRITER_UNKNOWN_THRESHOLD = 0.9

# Output paths
CKPT_PATH      = f"{OUTPUT_DIR}/baseline_{TASK}.pt"
BEST_CKPT_PATH = f"{OUTPUT_DIR}/baseline_{TASK}_best.pt"
LOG_PATH       = f"{OUTPUT_DIR}/log_{TASK}.json"

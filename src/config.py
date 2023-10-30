from pathlib import Path
import torch

DATA_DIR = Path(__file__).resolve().parents[1]/'input'/'captcha_images_v2'
BATCH_SIZE = 8
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75

NUM_WORKERS = 8
EPOCHS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    print(DEVICE)
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

TRAIN_DIR = ROOT_DIR / "data" / "input" / "chest_xray" / "chest_xray" / "train"
TEST_DIR  = ROOT_DIR / "data" / "input" / "chest_xray" / "chest_xray" / "test"

SAVED_MODEL_DIR   = ROOT_DIR / "saved_model"
CNN_MODEL_PATH    = SAVED_MODEL_DIR / "chest_xray_model.pt"
RESNET_MODEL_PATH = SAVED_MODEL_DIR / "resnet_model.pt"

IMG_HEIGHT = 180
IMG_WIDTH  = 180
BATCH_SIZE = 16

# ── Active training values ─────────────────────────────────────────────────────
EPOCHS        = 30
LEARNING_RATE = 1e-4    # best from PyTorch search (val_auc=0.9980)
DROPOUT       = 0.3
DENSE_UNITS   = 256
CNN_FILTERS   = 32

# Phase 2 ResNet fine-tuning
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LR     = 1e-5
FINE_TUNE_AT     = 100

VALIDATION_SPLIT = 0.2

# ── Search ranges (used by training/search.py) ────────────────────────────────
SEARCH = {
    "lr":      [1e-4, 1e-3, 5e-3],   # 3 learning rates
    "dropout": [0.3, 0.5],            # 2 dropout rates  →  3×2×2 = 12 combos
    "units":   [128, 256],            # 2 dense sizes
    "filters": [32],                  # fixed (64 OOMs on 6GB)
}

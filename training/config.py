from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

TRAIN_DIR = ROOT_DIR / "input" / "chest_xray" / "chest_xray" / "train"
VAL_DIR   = ROOT_DIR / "input" / "chest_xray" / "chest_xray" / "val"
TEST_DIR  = ROOT_DIR / "input" / "chest_xray" / "chest_xray" / "test"

SAVED_MODEL_DIR    = ROOT_DIR / "saved_model"
CNN_MODEL_PATH     = SAVED_MODEL_DIR / "chest_xray_model.keras"
RESNET_MODEL_PATH  = SAVED_MODEL_DIR / "resnet_model.keras"

IMG_HEIGHT = 180
IMG_WIDTH  = 180
BATCH_SIZE = 32
EPOCHS     = 3

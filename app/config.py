from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SAVED_MODEL_DIR = ROOT_DIR / "saved_model"
CNN_MODEL_PATH = SAVED_MODEL_DIR / "chest_xray_model.keras"
RESNET_MODEL_PATH = SAVED_MODEL_DIR / "resnet_model.keras"
README_PATH = ROOT_DIR / "README.md"

IMG_HEIGHT = 180
IMG_WIDTH = 180
PREDICTION_THRESHOLD = 0.8

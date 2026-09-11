import numpy as np
from PIL import Image

from app.config import IMG_HEIGHT, IMG_WIDTH


def preprocess_image(image_file) -> np.ndarray:
    img = Image.open(image_file).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

import tensorflow as tf
import numpy as np

from app.config import PREDICTION_THRESHOLD
from app.preprocessing.image import preprocess_image


def predict(model: tf.keras.Model, image_file) -> dict:
    img_array = preprocess_image(image_file)
    score = float(model.predict(img_array, verbose=0)[0][0])
    if score > PREDICTION_THRESHOLD:
        return {"label": "Pneumonia", "confidence": score}
    return {"label": "Normal", "confidence": 1.0 - score}

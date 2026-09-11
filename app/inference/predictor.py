import logging

import torch
import torch.nn as nn

from app.config import PREDICTION_THRESHOLD
from app.preprocessing.image import preprocess_image

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model: nn.Module, image_file) -> dict[str, float | str]:
    tensor = preprocess_image(image_file).to(DEVICE)

    with torch.no_grad():
        logit = model(tensor)
        pneumonia_prob = float(torch.sigmoid(logit).item())

    normal_prob = 1.0 - pneumonia_prob
    label       = "Pneumonia" if pneumonia_prob > PREDICTION_THRESHOLD else "Normal"
    confidence  = pneumonia_prob if label == "Pneumonia" else normal_prob

    log.debug(
        "Prediction — label: %s  pneumonia_prob: %.4f  confidence: %.4f",
        label, pneumonia_prob, confidence,
    )
    return {
        "label":        label,
        "confidence":   confidence,
        "pneumonia_pct": pneumonia_prob * 100,
        "normal_pct":    normal_prob * 100,
    }

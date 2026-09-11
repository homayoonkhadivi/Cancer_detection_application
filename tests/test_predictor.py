import io

import numpy as np
import pytest
from PIL import Image


def _make_image_bytes(width=100, height=100) -> io.BytesIO:
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_preprocess_output_shape():
    from app.preprocessing.image import preprocess_image

    result = preprocess_image(_make_image_bytes())
    assert result.shape == (1, 180, 180, 3)


def test_preprocess_values_normalized():
    from app.preprocessing.image import preprocess_image

    result = preprocess_image(_make_image_bytes())
    assert result.dtype == np.float32
    assert 0.0 <= result.min() and result.max() <= 1.0


def test_predict_returns_label_and_confidence():
    from unittest.mock import MagicMock

    import numpy as np

    from app.inference.predictor import predict

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.95]])

    result = predict(mock_model, _make_image_bytes())
    assert result["label"] == "Pneumonia"
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_normal_label():
    from unittest.mock import MagicMock

    import numpy as np

    from app.inference.predictor import predict

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.2]])

    result = predict(mock_model, _make_image_bytes())
    assert result["label"] == "Normal"

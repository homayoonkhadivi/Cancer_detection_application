import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.config import IMG_HEIGHT, IMG_WIDTH

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(image_file) -> torch.Tensor:
    img = Image.open(image_file).convert("RGB")
    return _TRANSFORM(img).unsqueeze(0)   # shape: (1, 3, H, W)

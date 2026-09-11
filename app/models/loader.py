import logging

import streamlit as st
import torch

from app.config import CNN_MODEL_PATH, RESNET_MODEL_PATH
from training.models import CNNModel, ResNetModel

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_cnn_model() -> CNNModel:
    model = CNNModel()
    model.load_state_dict(torch.load(str(CNN_MODEL_PATH), map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    log.info("CNN model loaded — path: %s  device: %s", CNN_MODEL_PATH, DEVICE)
    return model


@st.cache_resource
def load_resnet_model() -> ResNetModel:
    model = ResNetModel()
    model.load_state_dict(torch.load(str(RESNET_MODEL_PATH), map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    log.info("ResNet model loaded — path: %s  device: %s", RESNET_MODEL_PATH, DEVICE)
    return model

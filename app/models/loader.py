import streamlit as st
import tensorflow as tf

from app.config import CNN_MODEL_PATH, RESNET_MODEL_PATH


@st.cache_resource
def load_cnn_model() -> tf.keras.Model:
    return tf.keras.models.load_model(str(CNN_MODEL_PATH))


@st.cache_resource
def load_resnet_model() -> tf.keras.Model:
    return tf.keras.models.load_model(str(RESNET_MODEL_PATH))

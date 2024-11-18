import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Paths to models
CNN_MODEL_PATH = "./saved_model/chest_xray_model.keras"
RESNET_MODEL_PATH = "./saved_model/resnet_model.keras"

# Load models
@st.cache_resource
def load_model(model_path):
    """Load and return a trained model from the given path."""
    return tf.keras.models.load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image, img_height=180, img_width=180):
    """Preprocess uploaded image for prediction."""
    img = load_img(image, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Prediction function
def predict_image(model, image):
    """Make prediction using the specified model."""
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction[0][0]

# App UI
def main():
    st.title("Pneumonia Detection App")
    st.markdown(
        """
        **Welcome to the Pneumonia Detection App!**
        This app uses trained deep learning models (CNN and ResNet) to detect pneumonia from chest X-ray images. 
        Please select a model and upload an image to get the prediction.
        """
    )

    # Model selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.radio(
        "Select a model to use for prediction:",
        ("CNN Model", "ResNet Transfer Learning Model")
    )

    # Upload image
    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        st.write("")

        # Load selected model
        if model_choice == "CNN Model":
            model = load_model(CNN_MODEL_PATH)
            st.sidebar.write("Using CNN Model.")
        else:
            model = load_model(RESNET_MODEL_PATH)
            st.sidebar.write("Using ResNet Model.")

        # Make prediction
        if st.button("Predict"):
            with st.spinner("Processing..."):
                prediction = predict_image(model, uploaded_file)
                if prediction > 0.8:
                    st.success(f"The model predicts Pneumonia with a probability of {prediction:.2f}.")
                else:
                    st.success(f"The model predicts Normal with a probability of {1 - prediction:.2f}.")

    # Information from README
    st.sidebar.header("About this App")
    readme_path = "./README.md"  # Adjust path if needed
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            st.sidebar.markdown(f.read())
    else:
        st.sidebar.write("README file not found.")

if __name__ == "__main__":
    main()

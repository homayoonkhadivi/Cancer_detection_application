import streamlit as st

from app.config import README_PATH
from app.inference.predictor import predict
from app.models.loader import load_cnn_model, load_resnet_model

_MODEL_OPTIONS = ("CNN Model", "ResNet Transfer Learning Model")


def main() -> None:
    st.title("Pneumonia Detection App")
    st.markdown(
        "Upload a chest X-ray image and select a model to detect pneumonia."
    )

    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.radio("Select a model:", _MODEL_OPTIONS)

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        model = load_cnn_model() if model_choice == "CNN Model" else load_resnet_model()
        st.sidebar.caption(f"Using {model_choice}.")

        if st.button("Predict"):
            with st.spinner("Analysing..."):
                result = predict(model, uploaded_file)
            st.success(
                f"**{result['label']}** — confidence: {result['confidence']:.1%}"
            )

    _render_sidebar_readme()


def _render_sidebar_readme() -> None:
    st.sidebar.header("About")
    if README_PATH.exists():
        st.sidebar.markdown(README_PATH.read_text())

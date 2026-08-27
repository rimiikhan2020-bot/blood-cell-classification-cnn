"""
Blood Cell Classification — Streamlit Dashboard
Prepared by: Rimsha Pervaiz
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

try:
    import joblib
except ImportError:
    joblib = None

import tensorflow as tf

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
IMG_SIZE = 64
ARTIFACT_DIR = Path(__file__).parent / "assets" / "artifact"
MODEL_PATH = ARTIFACT_DIR / "blood_cell_cnn.keras"
ENCODER_PATH = ARTIFACT_DIR / "label_encoder.pkl"
LABELS_JSON_PATH = ARTIFACT_DIR / "labels.json"
DEFAULT_LABELS = ["RBC", "WBC"]

st.set_page_config(
    page_title="Blood Cell Classifier",
    page_icon="🩸",
    layout="centered",
)

# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_labels():
    if joblib is not None and ENCODER_PATH.exists():
        encoder = joblib.load(ENCODER_PATH)
        if hasattr(encoder, "classes_"):
            return list(encoder.classes_)
    if LABELS_JSON_PATH.exists():
        return json.loads(LABELS_JSON_PATH.read_text())
    return DEFAULT_LABELS


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🩸 Blood Cell Classification")
st.caption("CNN classifier for RBC vs WBC blood-smear crops · Rimsha Pervaiz")

model = load_model()
labels = load_labels()

if model is None:
    st.warning(
        f"No model found at `assets/artifact/{MODEL_PATH.name}`. "
        "Train the notebook and save the model there, then reload this app."
    )

with st.sidebar:
    st.header("About")
    st.write(
        "Classifies a cropped blood-cell image as **RBC** or **WBC** using a "
        "CNN trained on the Blood Cell Detection Dataset."
    )
    st.divider()
    st.subheader("Reported test performance")
    st.metric("Test accuracy", "99.72%")
    st.metric("Test loss", "0.0188")
    st.caption("RBC F1: 99.85% · WBC F1: 96.55%")

uploaded = st.file_uploader(
    "Upload a cropped blood-cell image (jpg / png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded crop", use_container_width=True)

    if model is not None:
        batch = preprocess(image)
        probs = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
        confidence = float(np.max(probs)) * 100

        with col2:
            st.subheader("Prediction")
            st.markdown(f"### {pred_label}")
            st.progress(min(int(confidence), 100))
            st.write(f"Confidence: **{confidence:.1f}%**")

            with st.expander("Class probabilities"):
                for label, p in zip(labels, probs):
                    st.write(f"{label}: {p * 100:.2f}%")
else:
    st.info("Upload a blood-cell crop above to classify it.")
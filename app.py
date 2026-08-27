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

CLASS_COLORS = {"RBC": "#E2836A", "WBC": "#9B7CC4"}
DEFAULT_COLOR = "#C9A227"

st.set_page_config(
    page_title="Blood Cell Classifier",
    page_icon="🩸",
    layout="centered",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

        .block-container { padding-top: 2.5rem; max-width: 880px; }

        h1, h2, h3 { font-family: 'Fraunces', serif !important; }

        .app-eyebrow {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #C9A227;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 6px;
        }
        .app-eyebrow::before { content: ""; width: 22px; height: 1px; background: #C9A227; display: inline-block; }

        .app-title {
            font-family: 'Fraunces', serif;
            font-weight: 700;
            font-size: 40px;
            color: #F4EFE4;
            margin: 0;
            line-height: 1.1;
        }
        .app-subtitle {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 13.5px;
            color: #A6A294;
            margin-top: 10px;
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: #1C202A;
            border: 1px dashed #3A3F4F;
            border-radius: 10px;
        }

        .pred-card {
            background: #1C202A;
            border: 1px solid #2C313E;
            border-radius: 10px;
            padding: 26px 28px;
        }
        .pred-label-tag {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #6B6A63;
        }
        .pred-value {
            font-family: 'Fraunces', serif;
            font-weight: 700;
            font-size: 36px;
            margin: 6px 0 16px 0;
        }
        .conf-track {
            width: 100%;
            height: 8px;
            background: #2C313E;
            border-radius: 100px;
            overflow: hidden;
        }
        .conf-fill { height: 100%; border-radius: 100px; }
        .conf-text {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12.5px;
            color: #A6A294;
            margin-top: 8px;
        }

        .prob-row { display: flex; align-items: center; gap: 12px; margin: 10px 0; }
        .prob-name {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12px;
            width: 60px;
            color: #ECE8DE;
        }
        .prob-track {
            flex: 1;
            height: 10px;
            background: #2C313E;
            border-radius: 100px;
            overflow: hidden;
        }
        .prob-fill { height: 100%; border-radius: 100px; }
        .prob-pct {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12px;
            color: #A6A294;
            width: 46px;
            text-align: right;
        }

        .sidebar-metric {
            background: #1C202A;
            border: 1px solid #2C313E;
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }
        .sidebar-metric .k {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6B6A63;
        }
        .sidebar-metric .v {
            font-family: 'Fraunces', serif;
            font-weight: 700;
            font-size: 24px;
            color: #F4EFE4;
            margin-top: 2px;
        }

        footer, #MainMenu { visibility: hidden; }
        .app-footer {
            margin-top: 48px;
            padding-top: 18px;
            border-top: 1px solid #2C313E;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11.5px;
            color: #6B6A63;
        }
    </style>
    """,
    unsafe_allow_html=True,
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
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="app-eyebrow">CNN · Blood-Smear Classifier</div>', unsafe_allow_html=True)
st.markdown('<h1 class="app-title">🩸 Blood Cell Classification</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Classifies a cropped blood-cell image as RBC or WBC · Prepared by Rimsha Pervaiz</div>',
    unsafe_allow_html=True,
)
st.write("")

model = load_model()
labels = load_labels()

if model is None:
    st.warning(
        f"No model found at `assets/artifact/{MODEL_PATH.name}`. "
        "Train the notebook and save the model there, then reload this app."
    )

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="app-eyebrow">About</div>', unsafe_allow_html=True)
    st.write(
        "Classifies a cropped blood-cell image as **RBC** or **WBC** using a "
        "CNN trained on the Blood Cell Detection Dataset."
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="app-eyebrow">Reported test performance</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-metric"><div class="k">Test accuracy</div><div class="v">99.72%</div></div>
        <div class="sidebar-metric"><div class="k">Test loss</div><div class="v">0.0188</div></div>
        <div class="sidebar-metric"><div class="k">RBC / WBC F1</div><div class="v" style="font-size:18px;">99.85% · 96.55%</div></div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Upload + prediction
# --------------------------------------------------------------------------
uploaded = st.file_uploader("Upload a cropped blood-cell image (jpg / png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded crop", use_container_width=True)

    if model is not None:
        batch = preprocess(image)
        probs = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
        confidence = float(np.max(probs)) * 100
        pred_color = CLASS_COLORS.get(pred_label, DEFAULT_COLOR)

        with col2:
            prob_rows = "".join(
                f"""
                <div class="prob-row">
                    <div class="prob-name">{label}</div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{p*100:.1f}%; background:{CLASS_COLORS.get(label, DEFAULT_COLOR)};"></div>
                    </div>
                    <div class="prob-pct">{p*100:.1f}%</div>
                </div>
                """
                for label, p in zip(labels, probs)
            )

            st.markdown(
                f"""
                <div class="pred-card">
                    <div class="pred-label-tag">Prediction</div>
                    <div class="pred-value" style="color:{pred_color};">{pred_label}</div>
                    <div class="conf-track">
                        <div class="conf-fill" style="width:{confidence:.1f}%; background:{pred_color};"></div>
                    </div>
                    <div class="conf-text">Confidence: {confidence:.1f}%</div>
                    <div style="margin-top:22px; border-top:1px solid #2C313E; padding-top:16px;">
                        <div class="pred-label-tag" style="margin-bottom:8px;">Class probabilities</div>
                        {prob_rows}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.info("Upload a blood-cell crop above to classify it.")

st.markdown(
    '<div class="app-footer">Blood Cell Detection Dataset · Kaggle (draaslan) &nbsp;·&nbsp; TensorFlow / Keras CNN</div>',
    unsafe_allow_html=True,
)

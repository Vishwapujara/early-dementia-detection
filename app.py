import os
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Early Dementia Detection",
    page_icon="🧠",
    layout="centered"
)

# ── Hugging Face repo — replace with your actual username ─────────────────
HF_REPO = "YOUR_HF_USERNAME/dementia-detection-models"

# ── Class labels ─────────────────────────────────────────────────────────
CNN_CLASSES = [
    "Non Demented",
    "Very Mild Dementia",
    "Mild Dementia",
    "Moderate Dementia"
]

# YOLO sorts alphabetically: 0=Mild, 1=Moderate, 2=Non Demented, 3=Very mild
# Remap to CNN order:        2      3              0               1
YOLO_TO_CNN = {0: 2, 1: 3, 2: 0, 3: 1}

CLASS_COLORS = {
    "Non Demented":       "#2ecc71",
    "Very Mild Dementia": "#f1c40f",
    "Mild Dementia":      "#e67e22",
    "Moderate Dementia":  "#e74c3c"
}

# ── Load models (cached — only downloads once per session) ────────────────
@st.cache_resource
def load_cnn():
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Dense, Conv2D

    with st.spinner("Downloading CNN model from Hugging Face..."):
        model_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="dementia_detection_model_final.h5"
        )

    # Handle Keras version mismatch (quantization_config argument)
    class CompatDense(Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)

    class CompatConv2D(Conv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)

    return load_model(
        model_path,
        custom_objects={'Dense': CompatDense, 'Conv2D': CompatConv2D}
    )

@st.cache_resource
def load_yolo():
    with st.spinner("Downloading YOLO model from Hugging Face..."):
        model_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="best.pt"
        )
    return YOLO(model_path)

cnn_model  = load_cnn()
yolo_model = load_yolo()

# ── Preprocessing ─────────────────────────────────────────────────────────
def preprocess_for_cnn(pil_image):
    img = pil_image.resize((128, 128)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_yolo(pil_image):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_image.save(tmp.name)
        tmp_path = tmp.name
    try:
        result     = yolo_model(tmp_path, imgsz=128, verbose=False)[0]
        yolo_class = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        cnn_class  = YOLO_TO_CNN[yolo_class]
        return cnn_class, confidence
    finally:
        os.unlink(tmp_path)

# ── UI ────────────────────────────────────────────────────────────────────
st.title("🧠 Early Dementia Detection")
st.write("Upload a brain MRI scan to classify Alzheimer's severity stage.")
st.write("Both models run independently — high agreement validates the diagnosis.")

uploaded_file = st.file_uploader(
    "Upload MRI Image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Scan", use_column_width=True)
    st.divider()

    with st.spinner("Running inference..."):
        arr           = preprocess_for_cnn(img)
        cnn_probs     = cnn_model.predict(arr, verbose=0)[0]
        cnn_class_idx = int(np.argmax(cnn_probs))
        cnn_conf      = float(np.max(cnn_probs))
        cnn_label     = CNN_CLASSES[cnn_class_idx]

        yolo_class_idx, yolo_conf = predict_yolo(img)
        yolo_label = CNN_CLASSES[yolo_class_idx]

    # ── Results side by side ──────────────────────────────────────────────
    st.subheader("Model Predictions")
    col1, col2 = st.columns(2)

    with col1:
        color = CLASS_COLORS[cnn_label]
        st.markdown("#### Custom 6-Block CNN")
        st.markdown(
            f"<div style='background:{color};padding:14px;border-radius:8px;"
            f"text-align:center;color:white;font-weight:bold;font-size:1.1em'>"
            f"{cnn_label}</div>",
            unsafe_allow_html=True
        )
        st.metric("Confidence", f"{cnn_conf*100:.1f}%")

    with col2:
        color = CLASS_COLORS[yolo_label]
        st.markdown("#### YOLOv8 Classifier")
        st.markdown(
            f"<div style='background:{color};padding:14px;border-radius:8px;"
            f"text-align:center;color:white;font-weight:bold;font-size:1.1em'>"
            f"{yolo_label}</div>",
            unsafe_allow_html=True
        )
        st.metric("Confidence", f"{yolo_conf*100:.1f}%")

    # ── Agreement indicator ───────────────────────────────────────────────
    st.divider()
    if cnn_class_idx == yolo_class_idx:
        st.success(f"✅ Both models agree: **{cnn_label}**")
    else:
        st.warning(
            f"⚠️ Models disagree — CNN: **{cnn_label}** | YOLO: **{yolo_label}**"
        )

    # ── CNN probability breakdown ─────────────────────────────────────────
    with st.expander("CNN class probabilities"):
        for i, prob in enumerate(cnn_probs):
            st.write(f"**{CNN_CLASSES[i]}:** {prob*100:.2f}%")
            st.progress(float(prob))

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write(
        "Classifies brain MRI scans into four Alzheimer's severity "
        "stages using two independently trained models."
    )
    st.write("**Severity stages:**")
    for name, color in CLASS_COLORS.items():
        st.markdown(
            f"<span style='color:{color};font-weight:bold'>■</span> {name}",
            unsafe_allow_html=True
        )
    st.divider()
    st.write("**CNN:** Custom 6-block, 4.46M params")
    st.write("**YOLO:** YOLOv8n-cls, 1.44M params")
    st.write("**Dataset:** OASIS — 86,437 MRI scans")
    st.write("**Benchmark agreement:** 77.46%")
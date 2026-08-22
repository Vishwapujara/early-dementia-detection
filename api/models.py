"""Model loading and inference for the FastAPI service.

Mirrors `app.py`'s Hugging Face Hub-based model loading (same repo, same
weight filenames), but derives the YOLO class-index -> class-name mapping
from the loaded model's own `.names` dict instead of a hardcoded table
(`app.py`'s `YOLO_TO_CNN`), so it stays correct even if a differently
class-ordered checkpoint gets deployed later.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_pipeline import CATEGORIES

HF_REPO = "Vishwapujara/dementia-detection-models"
CLASS_NAMES = list(CATEGORIES.keys())  # index-aligned with the ordinal label used everywhere else in the project


def load_cnn():
    from tensorflow.keras.layers import Conv2D, Dense
    from tensorflow.keras.models import load_model

    model_path = hf_hub_download(repo_id=HF_REPO, filename="dementia_detection_model_final.h5")

    # Handle Keras version mismatch (quantization_config argument) -- same fix as app.py
    class CompatDense(Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    class CompatConv2D(Conv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    return load_model(model_path, custom_objects={"Dense": CompatDense, "Conv2D": CompatConv2D})


def load_yolo():
    from ultralytics import YOLO

    model_path = hf_hub_download(repo_id=HF_REPO, filename="best.pt")
    return YOLO(model_path)


def predict_cnn(cnn_model, pil_image):
    img = pil_image.resize((128, 128)).convert("RGB")
    arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    probs = cnn_model.predict(arr, verbose=0)[0]
    class_idx = int(np.argmax(probs))
    return CLASS_NAMES[class_idx], float(probs[class_idx])


def predict_yolo(yolo_model, pil_image):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_image.save(tmp.name)
        tmp_path = tmp.name
    try:
        result = yolo_model(tmp_path, imgsz=128, verbose=False)[0]
        class_name = yolo_model.names[int(result.probs.top1)]
        confidence = float(result.probs.top1conf)
        return class_name, confidence
    finally:
        Path(tmp_path).unlink(missing_ok=True)

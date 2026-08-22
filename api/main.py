"""FastAPI inference service for the dementia detection models.

Exposes both the CNN and YOLOv8 models behind a single `/predict` endpoint
(the API counterpart to `app.py`'s dual-model Streamlit demo), logs every
prediction to SQLite (`api/database.py`), and exposes `/drift` for
Population-Stability-Index-based output-distribution monitoring
(`api/drift.py`) -- the deployment-time counterpart to the notebooks'
training-time benchmarking.
"""
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from api import models as model_utils
from api.database import get_recent_predictions, init_db, log_prediction
from api.drift import get_drift_status

_state = {"cnn": None, "yolo": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _state["cnn"] = model_utils.load_cnn()
    _state["yolo"] = model_utils.load_yolo()
    yield
    _state.clear()


app = FastAPI(title="Early Dementia Detection API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {"cnn": _state["cnn"] is not None, "yolo": _state["yolo"] is not None},
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _state["cnn"] is None or _state["yolo"] is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet")

    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file as an image")

    cnn_class, cnn_conf = model_utils.predict_cnn(_state["cnn"], image)
    yolo_class, yolo_conf = model_utils.predict_yolo(_state["yolo"], image)

    log_prediction("cnn", cnn_class, cnn_conf)
    log_prediction("yolo", yolo_class, yolo_conf)

    return {
        "cnn": {"class": cnn_class, "confidence": round(cnn_conf, 4)},
        "yolo": {"class": yolo_class, "confidence": round(yolo_conf, 4)},
        "agree": cnn_class == yolo_class,
    }


@app.get("/predictions/recent")
def recent_predictions(limit: int = 50, model: Optional[str] = None):
    return get_recent_predictions(limit=limit, model=model)


@app.get("/drift")
def drift(model: Optional[str] = None, limit: int = 200):
    if model is not None:
        return get_drift_status(model, limit=limit)
    return {
        "cnn": get_drift_status("cnn", limit=limit),
        "yolo": get_drift_status("yolo", limit=limit),
    }

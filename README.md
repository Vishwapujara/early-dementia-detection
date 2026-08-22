# Early Dementia Detection from Brain MRI

Ordinal classification of Alzheimer's severity stages using a custom CNN, YOLOv8, and a staged-transfer-learning DenseNet121, trained on 86K MRI scans from the OASIS dataset, deployed behind a FastAPI service with prediction logging and drift monitoring.

## Results

| Model | Accuracy | Scott's Pi | QWK |
|-------|----------|------------|-----|
| Custom 6-Block CNN | 75.73% | 0.4257 | 0.6680 |
| YOLOv8 | 91.64% | 0.7869 | 0.8447 |
| DenseNet121 (staged transfer learning) | *pending* | *pending* | *pending* |

**Prediction Agreement: 77.46%** — the CNN and YOLOv8 independently agreed on the same diagnosis 3 out of 4 times despite being built on completely different architectures. See `notebooks/densenet_benchmark.ipynb` for the DenseNet121 benchmark methodology; it needs a GPU run against the full OASIS dataset to fill in real numbers.

## Project Structure

```
├── notebooks/
│   ├── Dementia.ipynb            # CNN training notebook (Google Colab)
│   ├── yolo_benchmark.ipynb      # YOLOv8 training + benchmark notebook
│   └── densenet_benchmark.ipynb  # DenseNet121 staged transfer learning notebook
├── src/
│   ├── data_pipeline.py      # Metadata extraction, patient-level splitting, resampling (pandas)
│   ├── spark_pipeline.py     # Same pipeline on PySpark's DataFrame API
│   └── evaluate.py           # Scott's Pi / Quadratic Weighted Kappa metrics
├── scripts/
│   └── verify_pipeline_parity.py  # Proves data_pipeline.py and spark_pipeline.py agree
├── api/                       # FastAPI inference service (prediction logging + drift monitoring)
│   ├── main.py
│   ├── models.py
│   ├── database.py
│   └── drift.py
├── models/                   # best.pt, dementia_detection_model_final.h5 (not tracked, see note below)
├── app.py                    # Streamlit web app
├── Dockerfile                 # Containerizes the FastAPI service
├── requirements.txt          # Dependencies (notebooks, Streamlit app)
├── requirements-api.txt      # Dependencies (FastAPI service / Docker image)
└── README.md
```

## Four Severity Stages

| Stage | Description |
|-------|-------------|
| Non Demented | No signs of Alzheimer's |
| Very Mild Dementia | Early subtle changes |
| Mild Dementia | Noticeable cognitive decline |
| Moderate Dementia | Significant impairment |

## Key Technical Decisions

**Ordinal classification** — labels encoded as integers (0,1,2,3) preserving severity order, trained with sparse categorical crossentropy.

**Patient-level splitting** — data split by patient ID not by image, preventing leakage where the same patient's scans appear in both train and test.

**Hybrid resampling** — 137:1 class imbalance (67K Non Demented vs 488 Moderate) handled by undersampling majority classes and oversampling minority classes to 8000 per class.

**Multi-model benchmark** — CNN, YOLOv8, and DenseNet121 trained independently on the same patient-level splits, then compared on the same test set to validate diagnostic consistency.

**PySpark pipeline parity** — `src/spark_pipeline.py` reimplements metadata extraction, splitting, and resampling on Spark's DataFrame API; `scripts/verify_pipeline_parity.py` proves it produces identical patient splits and class distributions to the pandas version, so either can be used interchangeably as the dataset scales.

## CNN Architecture

Custom 6-block CNN built from scratch (4.46M parameters):

```
Block 1:  Conv2D(32)  → BatchNorm → MaxPool → Dropout
Block 2:  Conv2D(64)  → BatchNorm → MaxPool → Dropout
Block 3:  Conv2D(128) → BatchNorm → MaxPool → Dropout
Block 4:  Conv2D(256) → BatchNorm → MaxPool → Dropout
Block 5:  Conv2D(512) → BatchNorm → MaxPool → Dropout
Block 6:  Conv2D(512) → BatchNorm → MaxPool → Dropout
          Flatten → Dense(256) → Dropout → Dense(4, softmax)
```

## Dataset

OASIS (Open Access Series of Imaging Studies)
- 86,437 MRI brain scans along the z-axis
- 461 participants
- 4 severity classes with 137:1 imbalance

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

The app loads both models and lets you upload an MRI scan to get predictions from both models side by side.

> **Note:** Model weights (`dementia_detection_model_final.h5` and `best.pt`) are not included in this repository due to file size. Download them separately and place them in `models/`.

## API Deployment

A FastAPI service (`api/`) exposes both the CNN and YOLOv8 models behind a single `/predict` endpoint — the deployment counterpart to the Streamlit demo. Every prediction is logged to SQLite, and a `/drift` endpoint reports each model's recent output distribution against the real training-set baseline using the Population Stability Index (PSI).

```bash
# Locally
pip install -r requirements-api.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Or containerized
docker build -t dementia-api .
docker run -p 8000:8000 dementia-api
```

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Service + model-loaded status |
| `POST /predict` | Upload an MRI scan, get both models' predictions and whether they agree |
| `GET /predictions/recent` | Recently logged predictions (`?limit=&model=`) |
| `GET /drift` | PSI-based drift status per model against the training-set baseline (`?model=&limit=`) |

Both models download from Hugging Face Hub at startup (same repo `app.py` uses), so no local weight files are needed to run the container.

## Tech Stack

- Python, TensorFlow, Keras
- PyTorch, Ultralytics YOLOv8
- FastAPI, Docker
- PySpark
- Streamlit = https://early-dementia-detection-lbbyrgmjx2xa945rpmu9js.streamlit.app/
- scikit-learn, pandas, numpy, matplotlib

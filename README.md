# Early Dementia Detection from Brain MRI

Ordinal classification of Alzheimer's severity stages using a custom CNN and YOLOv8, trained on 86K MRI scans from the OASIS dataset.

## Results

| Model | Accuracy | Scott's Pi | QWK |
|-------|----------|------------|-----|
| Custom 6-Block CNN | 75.73% | 0.4257 | 0.6680 |
| YOLOv8 | 91.64% | 0.7869 | 0.8447 |

**Prediction Agreement: 77.46%** — both models independently agreed on the same diagnosis 3 out of 4 times despite being built on completely different architectures.

## Project Structure

```
├── Dementia.ipynb            # CNN training notebook (Google Colab)
├── yolo_benchmark.ipynb      # YOLOv8 training + benchmark notebook
├── app.py                    # Streamlit web app
├── requirements.txt          # Dependencies
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

**Dual-model benchmark** — CNN and YOLOv8 trained independently on the same dataset, then compared on the same test set to validate diagnostic consistency.

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

> **Note:** Model weights (`dementia_detection_model_final.h5` and `best.pt`) are not included in this repository due to file size. Download them separately and place them in the project root.

## Tech Stack

- Python, TensorFlow, Keras
- PyTorch, Ultralytics YOLOv8
- Streamlit
- scikit-learn, pandas, numpy, matplotlib
# Early Dementia Detection — inference API
# Builds a container serving api/main.py (FastAPI) with the CNN + YOLOv8
# models downloaded from Hugging Face Hub at startup (see api/models.py).
FROM python:3.11-slim

WORKDIR /app

# libgl1/libglib2.0-0: ultralytics pulls in opencv-python, which needs
# libGL.so.1 -- not present in the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY src/ ./src/

# predictions.db is written to /app at runtime (api/database.py resolves it
# relative to the api/ package) -- mount a volume at /app if you want
# prediction history to survive container restarts:
#   docker run -v dementia_data:/app -p 8000:8000 <image>
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional optimized packages
RUN pip install --no-cache-dir \
    uvloop \
    httptools \
    orjson

# Copy application code
COPY . .

# Make scripts executable
RUN chmod +x *.py

# Set environment variables
ENV PORT=8000
ENV TF_SERVING_HOST=vocal-model-service
ENV TF_SERVING_PORT=8501
ENV MODEL_ENDPOINT=http://vocal-model-service:8501/v1/models/vocal_transformation_model:predict
ENV USE_LOCAL_MODEL=false

# Expose port
EXPOSE 8000

# Run server with performance optimizations
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--loop", "uvloop", "--http", "httptools"]
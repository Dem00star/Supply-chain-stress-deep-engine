# 1. Base Image: We use Python 3.12 'slim' to keep the cloud server lightweight and fast
FROM python:3.12-slim

# 2. Set the working directory inside the virtual Linux machine
WORKDIR /app

# 3. Install core system dependencies that PyTorch and Pandas sometimes need on bare Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy ONLY the requirements file first. 
# (This is an MLOps trick: Docker caches this step, so if you change a python script later, 
# it doesn't waste 10 minutes reinstalling PyTorch from scratch).
COPY backend_requirements.txt .

# 5. Install the Python libraries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend_requirements.txt

# 6. Copy your actual architecture into the container
COPY src/api/ /app/src/api/
COPY src/models/ /app/src/models/
COPY models/ /app/models/
COPY data/processed/master_feature_set.parquet /app/data/processed/

# 7. Open port 8000 so the container can receive traffic from Streamlit
EXPOSE 8000

# 8. The Boot Command: Start the FastAPI server on 0.0.0.0 so the cloud provider can route traffic to it
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
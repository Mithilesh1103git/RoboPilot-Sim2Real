# NVIDIA CUDA runtime image for GPU-capable VM/container deployment.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
COPY training/requirements-training.txt /workspace/training/requirements-training.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r /workspace/requirements.txt -r /workspace/training/requirements-training.txt boto3
COPY app /workspace/app
COPY training /workspace/training
COPY configs /workspace/configs
COPY assets /workspace/assets
EXPOSE 8080
CMD ["python3", "-m", "uvicorn", "training.deploy.service:app", "--host", "0.0.0.0", "--port", "8080"]

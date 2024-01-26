FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-11.6

# Install necessary packages and Python
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3-pip python3.9-dev build-essential git

# Upgrade pip
RUN python3.9 -m pip install --upgrade pip

WORKDIR /instructions

# Copy your application
COPY . /instructions

# Install requirements
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt

# Install PyTorch compatible with CUDA 11.6
RUN python3.9 -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116

# Run your script
CMD ["python3.9", "llama2/scripts/finetune.py"]

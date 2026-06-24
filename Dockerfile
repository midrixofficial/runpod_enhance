# Use a CUDA runtime base image with Python support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Avoid prompt questions during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Set working directory inside container
WORKDIR /app

# Copy requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ONNX model and the RunPod handler
COPY realesr_general_x4v3.onnx .
COPY handler.py .

# Run the handler (unbuffered output for better logging)
CMD ["python3", "-u", "handler.py"]

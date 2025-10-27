FROM ultralytics/ultralytics:latest-jetson-jetpack6

# Set working directory
WORKDIR /app

# Install system dependencies including curl for uv installation
RUN apt-get update && apt-get install -y vim busybox curl && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Jetson.GPIO with older setuptools to avoid compatibility issues
RUN python3 -m pip install --upgrade pip && \
    pip install "setuptools==69.5.1" && \
    pip install Jetson.GPIO && \
    pip install "setuptools<75.0.0"

# Copy dependency files first (for better Docker cache usage)
COPY pyproject.toml ./

# Install ONLY the additional dependencies we need for Jetson deployment
# DO NOT reinstall ultralytics, pytorch, opencv, numpy - they come with the base image!
# The base image (ultralytics/ultralytics:latest-jetson-jetpack6) already has:
#   - PyTorch with CUDA support for Jetson
#   - Ultralytics YOLO
#   - OpenCV, numpy, pillow
# We only add: fastapi, uvicorn, lerobot, feetech-servo-sdk, supervision, pyyaml
RUN pip install \
    fastapi>=0.110.3 \
    uvicorn \
    supervision \
    lerobot>=0.3.0 \
    feetech-servo-sdk>=1.0.0 \
    pyyaml

# Copy configuration file
COPY config.yaml ./

# Copy shared inference module (required by inference.py and rt_200.py)
COPY yolo_inference.py ./

# Copy project files (after pip install so code changes don't invalidate pip cache)
COPY inference.py ./
COPY rt_200.py ./
COPY test_camera.py ./
COPY csi_camera_capture.py ./
COPY servo_test_sysfs.py ./
COPY trigger_position_test.py ./
COPY pitch_test.py ./
COPY find_motors.py ./

# Copy model files
COPY *.pt ./

# Copy runs directory with trained models
COPY runs ./runs

# Copy static files directory
COPY static ./static

# Create detections directory for rt_200.py
RUN mkdir -p detections

# Expose port for web server
EXPOSE 8000

# Default command runs the rt_200.py server
# Users can override this to run inference.py instead
CMD ["python3", "rt_200.py", "--enable-camera", "--use-csi", "--no-connect", "--model", "runs/yolo11n-2025-10-24/weights/best.pt", "--imgsz", "640"]

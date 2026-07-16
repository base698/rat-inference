FROM ultralytics/ultralytics:latest-jetson-jetpack6

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y vim busybox && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Install Jetson.GPIO with older setuptools to avoid compatibility issues
RUN python3 -m pip install --upgrade pip && \
    pip install "setuptools==69.5.1" && \
    pip install Jetson.GPIO && \
    pip install "setuptools<75.0.0" && \
    pip install \
        "numpy<2" \
        "pillow>=10.0.0" \
        opencv-python \
        supervision \
        "fastapi>=0.110.3" \
        uvicorn \
        pyserial \
        deepdiff \
        accelerate \
        datasets \
        feetech-servo-sdk && \
    pip install --no-deps lerobot
# Copy project files (after pip install so code changes don't invalidate pip cache)
COPY config.yaml ./
COPY pyproject.toml ./
COPY rt_200.py ./
COPY ratbot ./ratbot
COPY tools ./tools

# Copy runs directory with trained models
COPY runs ./runs

# Copy static files directory
COPY static ./static

# Create detections directory for rt_200.py
RUN mkdir -p detections

# Expose port for web server
EXPOSE 8000

# Default command runs the rt_200.py server with stereo mode
# Users can override this to run tools/vision/inference/inference.py instead
CMD ["python3", "rt_200.py", "--enable-camera", "--use-csi", "--no-connect", "--stereo", "--model", "runs/yolo11n-2025-10-23/weights/best.engine"]

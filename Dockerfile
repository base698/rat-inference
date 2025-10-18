FROM ultralytics/ultralytics:latest-jetson-jetpack6

# Set working directory
WORKDIR /app

# Install Python dependencies first (so Docker caches this layer)
# Note: Only installing packages that are actually used by the code
# Excluded: lerobot, feetech-servo-sdk (servo control), inference, ncnn, onnx (not imported)

# Install Jetson.GPIO from GitHub (PyPI version has setuptools compatibility issues)
RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/NVIDIA/jetson-gpio.git /tmp/jetson-gpio && \
    cd /tmp/jetson-gpio && \
    python3 setup.py install && \
    cd / && rm -rf /tmp/jetson-gpio

# Install other Python dependencies
RUN python3 -m pip install --upgrade pip && \
    pip install "setuptools<75.0.0" && \
    pip install \
        "numpy<2" \
        pillow>=10.0.0 \
        ultralytics>=8.3.179 \
        opencv-python \
        supervision \
        fastapi>=0.110.3 \
        uvicorn

RUN apt update
RUN apt install -y vim
# Copy project files (after pip install so code changes don't invalidate pip cache)
COPY pyproject.toml ./
COPY inference.py ./
COPY rt_200.py ./
COPY test_camera.py ./
COPY csi_camera_capture.py ./
COPY servo_test.py ./

# Copy model files
COPY *.pt ./

# Copy runs directory with trained models
COPY runs ./runs

# Create detections directory for rt_200.py
RUN mkdir -p detections

# Expose port for web server
EXPOSE 8000

# Default command runs the rt_200.py server
# Users can override this to run inference.py instead
CMD ["python3", "rt_200.py", "--enable-camera", "--use-csi", "--no-connect"]

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
        pillow>=10.0.0 \
        ultralytics>=8.3.179 \
        opencv-python \
        supervision \
        fastapi>=0.110.3 \
        uvicorn
# Copy project files (after pip install so code changes don't invalidate pip cache)
COPY pyproject.toml ./
COPY inference.py ./
COPY rt_200.py ./
COPY test_camera.py ./
COPY csi_camera_capture.py ./
COPY servo_test.py ./
COPY servo_test_sysfs.py ./
COPY find_pwm.py ./
COPY test_all_pwm.py ./

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

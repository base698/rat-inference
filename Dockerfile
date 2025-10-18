FROM ultralytics/ultralytics:latest-jetson-jetpack6

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY inference.py ./
COPY rt_200.py ./

# Copy model files
COPY *.pt ./

# Copy runs directory with trained models
COPY runs ./runs

# Install Python dependencies
# Note: Only installing packages that are actually used by the code
# Excluded: lerobot, feetech-servo-sdk (servo control), inference, ncnn, onnx (not imported)
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

# Create detections directory for rt_200.py
RUN mkdir -p detections

# Expose port for web server
EXPOSE 8000

# Default command runs the rt_200.py server
# Users can override this to run inference.py instead
CMD ["python3", "rt_200.py", "--enable-camera", "--use-csi", "--no-connect"]

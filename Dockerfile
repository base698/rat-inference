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

# Install Python dependencies from pyproject.toml
# Note: servo libraries (lerobot, feetech-servo-sdk) are excluded due to compatibility issues
# The code handles their absence gracefully with --no-connect flag
RUN python3 -m pip install --upgrade pip && \
    pip install \
        pillow>=10.0.0 \
        ultralytics>=8.3.179 \
        numpy \
        opencv-python \
        "inference>=0.35.0,<0.51" \
        supervision \
        fastapi>=0.110.3 \
        uvicorn \
        "ncnn>=1.0.20250916" \
        "onnx>=1.19.0"

# Create detections directory for rt_200.py
RUN mkdir -p detections

# Expose port for web server
EXPOSE 8000

# Default command runs the rt_200.py server
# Users can override this to run inference.py instead
CMD ["python3", "rt_200.py", "--enable-camera", "--use-csi", "--no-connect"]

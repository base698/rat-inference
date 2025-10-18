FROM ultralytics/ultralytics:latest-jetson-jetpack6

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY inference.py ./
COPY rt_200.py ./

# Copy model files if they exist
COPY *.pt ./

# Copy any runs directory with trained models
COPY runs/ ./runs/ 2>/dev/null || true

# Install Python dependencies from pyproject.toml
# Extract dependencies and install them
RUN python3 -m pip install --upgrade pip && \
    python3 -c "import tomllib; deps = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; print('\n'.join(deps))" > /tmp/requirements.txt && \
    pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Create detections directory for rt_200.py
RUN mkdir -p detections

# Expose port for web server
EXPOSE 8000

# Default command runs the rt_200.py server
# Users can override this to run inference.py instead
CMD ["python3", "rt_200.py", "--enable-camera", "--use-csi", "--no-connect"]

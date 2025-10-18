#!/bin/bash

# Docker run script for rat-inference on Jetson Orin
# This script builds and runs the container with proper GPU and device access

IMAGE_NAME="rat-inference:latest"

# Function to display usage
usage() {
    echo "Usage: $0 [build|rt200|inference|shell]"
    echo ""
    echo "Commands:"
    echo "  build      - Build the Docker image"
    echo "  rt200      - Run rt_200.py server (default, with web UI on port 8000)"
    echo "  inference  - Run inference.py for testing"
    echo "  shell      - Open a bash shell in the container"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 rt200"
    echo "  $0 inference --input test.jpg --model yolo11n.pt --save"
    exit 1
}

# Build the image
build() {
    echo "Building Docker image: $IMAGE_NAME"
    docker build -t $IMAGE_NAME .
}

# Run rt_200.py with web server
run_rt200() {
    echo "Starting rt_200.py server..."
    echo "Web interface will be available at http://localhost:8000"
    docker run -it --rm \
        --ipc=host \
        --runtime=nvidia \
        --network=host \
        --device /dev/video0 \
        --device /dev/i2c-1 \
        -v $(pwd)/detections:/app/detections \
        -v $(pwd)/runs:/app/runs \
        $IMAGE_NAME \
        python3 rt_200.py --enable-camera --use-csi --no-connect "$@"
}

# Run inference.py
run_inference() {
    echo "Running inference.py..."
    docker run -it --rm \
        --ipc=host \
        --runtime=nvidia \
        -v $(pwd):/app/data \
        $IMAGE_NAME \
        python3 inference.py "$@"
}

# Open a shell in the container
run_shell() {
    echo "Opening shell in container..."
    docker run -it --rm \
        --ipc=host \
        --runtime=nvidia \
        --network=host \
        --device /dev/video0 \
        --device /dev/i2c-1 \
        -v $(pwd):/app/data \
        $IMAGE_NAME \
        bash
}

# Main script logic
case "${1:-rt200}" in
    build)
        build
        ;;
    rt200)
        shift
        run_rt200 "$@"
        ;;
    inference)
        shift
        run_inference "$@"
        ;;
    shell)
        run_shell
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Unknown command: $1"
        usage
        ;;
esac

#!/bin/bash

# Docker run script for rat-inference on Jetson Orin
# This script builds and runs the container with proper GPU and device access

IMAGE_NAME="rat-inference:latest"

# Function to display usage
usage() {
    echo "Usage: $0 [build|rt200|inference|shell|servo-test|trigger-test|pitch-test|motor-scan]"
    echo ""
    echo "Commands:"
    echo "  build          - Build the Docker image"
    echo "  rt200          - Run rt_200.py server (default, with web UI on port 8000)"
    echo "  inference      - Run inference.py for testing"
    echo "  test-cam       - Test camera and save test image"
    echo "  servo-test     - Test servo using hardware PWM (sysfs)"
    echo "  trigger-test   - Interactive trigger servo position test (type numbers, 'q' to quit)"
    echo "  pitch-test     - Test pitch motor (read/write position)"
    echo "  motor-scan     - Scan for Feetech motors and change IDs"
    echo "  shell          - Open a bash shell in the container"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 rt200"
    echo "  $0 inference --input ./test.jpg --save"
    echo "  $0 servo-test 90          # Move servo to 90 degrees"
    echo "  $0 servo-test             # Continuous sweep test"
    echo "  $0 trigger-test           # Interactive trigger position test"
    echo "  $0 pitch-test             # Read current pitch position"
    echo "  $0 pitch-test 300         # Write pitch position to 300"
    echo "  $0 motor-scan             # Scan for motors on IDs 1-20"
    echo "  $0 motor-scan --set-id 1 5  # Change motor from ID 1 to ID 5"
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
    docker run -d --rm \
        --name rat-inference \
        --privileged \
        --ipc=host \
        --runtime=nvidia \
        --network=host \
        --pid=host \
        -e DISPLAY=$DISPLAY \
        -v /tmp:/tmp \
        -v /var/run:/var/run \
        -v $(pwd)/detections:/app/detections \
        -v $(pwd)/runs:/app/runs \
        --device=/dev/ttyACM0:/dev/ttyACM0 \
        --group-add video \
        $IMAGE_NAME \
        python3 rt_200.py \
            --enable-camera \
            --port /dev/ttyACM0 \
            --model runs/yolo11n-2025-10-22/weights/best.pt \
            --use-csi \
            --camera-id 0 \
            "$@"

    echo ""
    echo "Container started in background with name: rat-inference"
    echo "To view logs: docker logs -f rat-inference"
    echo "To stop: docker stop rat-inference"
}

# Run inference.py
run_inference() {
    echo "Running inference.py..."

    # Convert local file paths to container paths
    ARGS=()
    for arg in "$@"; do
        if [[ "$arg" == --input=* ]]; then
            # Handle --input=file.jpg format
            path="${arg#--input=}"
            path="${path#./}"  # Remove leading ./
            ARGS+=("--input=/app/data/$path")
        elif [[ "$arg" == --output=* ]]; then
            # Handle --output=file.jpg format
            path="${arg#--output=}"
            path="${path#./}"
            ARGS+=("--output=/app/data/$path")
        elif [[ "$arg" == "-i" ]] || [[ "$arg" == "--input" ]] || [[ "$arg" == "-o" ]] || [[ "$arg" == "--output" ]]; then
            ARGS+=("$arg")
            # Next arg will be processed specially
            CONVERT_NEXT=true
        elif [[ "$CONVERT_NEXT" == true ]]; then
            # This is a path argument, convert it
            path="${arg#./}"  # Remove leading ./
            ARGS+=("/app/data/$path")
            CONVERT_NEXT=false
        else
            ARGS+=("$arg")
        fi
    done

    docker run -it --rm \
        --ipc=host \
        --runtime=nvidia \
        -v $(pwd):/app/data \
        $IMAGE_NAME \
        python3 inference.py \
            --model runs/yolo11n-2025-10-22/weights/best.pt \
            "${ARGS[@]}"
}

# Test camera
test_camera() {
    echo "Testing camera..."
    docker run -it --rm \
        --privileged \
        --ipc=host \
        --runtime=nvidia \
        -v /tmp/argus_socket:/tmp/argus_socket \
        -v $(pwd):/app/data \
        --group-add video \
        $IMAGE_NAME \
        python3 test_camera.py
}

# Test servo (sysfs hardware PWM method)
test_servo() {
    echo "Testing servo control (sysfs hardware PWM)..."
    echo "Make sure servo is connected to Pin 33 (GPIO13/PWM1)"
    docker run -it --rm \
        --privileged \
        --ipc=host \
        --runtime=nvidia \
        -v /sys:/sys \
        $IMAGE_NAME \
        python3 servo_test_sysfs.py "$@"
}

# Test pitch motor (Feetech)
test_pitch() {
    echo "Testing pitch motor (Feetech STS3215)..."
    docker run -it --rm \
        --privileged \
        --ipc=host \
        --runtime=nvidia \
        --device=/dev/ttyACM0:/dev/ttyACM0 \
        $IMAGE_NAME \
        python3 pitch_test.py "$@"
}

# Test trigger servo positions (interactive)
test_trigger() {
    echo "Testing trigger servo positions (interactive)..."
    echo "Make sure trigger servo is connected to Pin 15 (GPIO12/PWM0)"
    docker run -it --rm \
        --privileged \
        --ipc=host \
        --runtime=nvidia \
        -v /sys:/sys \
        $IMAGE_NAME \
        python3 trigger_position_test.py "$@"
}

# Scan for motors and change IDs
scan_motors() {
    echo "Scanning for Feetech motors..."
    docker run -it --rm \
        --privileged \
        --ipc=host \
        --runtime=nvidia \
        --device=/dev/ttyACM0:/dev/ttyACM0 \
        $IMAGE_NAME \
        python3 motor_scan.py "$@"
}

# Open a shell in the container
run_shell() {
    echo "Opening shell in container..."
    docker run -it --rm \
        --privileged \
        --ipc=host \
        --runtime=nvidia \
        --network=host \
        -v /tmp/argus_socket:/tmp/argus_socket \
        -v $(pwd):/app/data \
        --group-add video \
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
    test-cam)
        test_camera
        ;;
    servo-test)
        shift
        test_servo "$@"
        ;;
    pitch-test)
        shift
        test_pitch "$@"
        ;;
    trigger-test)
        shift
        test_trigger "$@"
        ;;
    motor-scan)
        shift
        scan_motors "$@"
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

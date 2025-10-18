#!/usr/bin/env python3
"""
Minimal camera test - takes a picture and saves it
"""
import cv2
import sys
from datetime import datetime

def test_usb_camera(camera_id=0):
    """Test USB camera (regular video device)"""
    print(f"Testing USB camera on device {camera_id}...")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Failed to open camera device {camera_id}")
        return False

    print("✓ Camera opened successfully")

    # Try to capture a frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame")
        cap.release()
        return False

    print(f"✓ Frame captured: {frame.shape}")

    # Save the frame
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/app/data/test_capture_usb_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"✓ Image saved: {filename}")

    cap.release()
    return True

def test_csi_camera(sensor_id=0):
    """Test CSI camera with GStreamer"""
    print(f"Testing CSI camera (sensor-id={sensor_id})...")

    # GStreamer pipeline for Jetson CSI camera
    pipeline = (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
        f"nvvidconv flip-method=2 ! "
        f"video/x-raw, width=640, height=480, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink"
    )

    print(f"Pipeline: {pipeline}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ Failed to open CSI camera")
        return False

    print("✓ CSI camera opened successfully")

    # Try to capture a frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame from CSI camera")
        cap.release()
        return False

    print(f"✓ Frame captured: {frame.shape}")

    # Save the frame
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/app/data/test_capture_csi_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"✓ Image saved: {filename}")

    cap.release()
    return True

def main():
    print("=" * 60)
    print("Camera Test Script")
    print("=" * 60)

    # Test USB camera first
    usb_success = test_usb_camera(0)
    print()

    # Test CSI camera
    csi_success = test_csi_camera(0)
    print()

    print("=" * 60)
    print("Results:")
    print(f"  USB Camera (device 0): {'✓ PASS' if usb_success else '❌ FAIL'}")
    print(f"  CSI Camera (sensor 0): {'✓ PASS' if csi_success else '❌ FAIL'}")
    print("=" * 60)

    if usb_success or csi_success:
        print("\nCheck /app/data/ for captured images")
        sys.exit(0)
    else:
        print("\nBoth camera tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

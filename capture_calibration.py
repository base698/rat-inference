#!/usr/bin/env python3
"""
Headless Camera Calibration Image Capture for Jetson
Captures calibration images via SSH with text feedback (no display needed)
"""

import cv2
import os
import argparse
import time
from datetime import datetime

def capture_calibration_images(camera_id, output_dir, pattern_size=(9, 6),
                               num_images=30, use_csi=False, stereo_mode=False):
    """
    Capture calibration images with text feedback for headless operation

    Args:
        camera_id: Camera device ID (or ignored if use_csi)
        output_dir: Directory to save images
        pattern_size: Checkerboard internal corners (width, height)
        num_images: Target number of images to capture
        use_csi: Use CSI camera with GStreamer
        stereo_mode: Capture from two cameras for stereo calibration
    """
    os.makedirs(output_dir, exist_ok=True)

    # Open camera(s)
    if use_csi:
        print("Opening CSI camera...")
        gst_pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        cap2 = None
        if stereo_mode:
            print("Opening second CSI camera...")
            gst_pipeline2 = gst_pipeline.replace("sensor-id=0", "sensor-id=1")
            cap2 = cv2.VideoCapture(gst_pipeline2, cv2.CAP_GSTREAMER)
    else:
        print(f"Opening USB camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap2 = None
        if stereo_mode:
            print(f"Opening second USB camera {camera_id + 1}...")
            cap2 = cv2.VideoCapture(camera_id + 1)
            cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    if stereo_mode and not cap2.isOpened():
        print("ERROR: Could not open second camera")
        return

    print("\n" + "="*60)
    print("CALIBRATION IMAGE CAPTURE (Headless Mode)")
    print("="*60)
    print(f"Pattern size: {pattern_size[0]}x{pattern_size[1]} internal corners")
    print(f"Target images: {num_images}")
    print(f"Output directory: {output_dir}")
    print("\nInstructions:")
    print("  - Move checkerboard to different positions and angles")
    print("  - Pattern must be fully visible in frame")
    print("  - Press ENTER to capture when pattern is detected")
    print("  - Type 'q' and ENTER to quit")
    print("="*60 + "\n")

    captured_count = 0

    # Create subdirectories for stereo
    if stereo_mode:
        os.makedirs(f"{output_dir}/left", exist_ok=True)
        os.makedirs(f"{output_dir}/right", exist_ok=True)

    while captured_count < num_images:
        # Capture frame(s)
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame")
            break

        ret2, frame2 = (cap2.read() if stereo_mode else (None, None))
        if stereo_mode and not ret2:
            print("ERROR: Failed to capture from second camera")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if stereo_mode else None

        # Find checkerboard corners
        found, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)

        found2 = False
        if stereo_mode and found:
            found2, corners2 = cv2.findChessboardCorners(gray2, pattern_size,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                         cv2.CALIB_CB_NORMALIZE_IMAGE)

        # Status update
        status = "✓ PATTERN DETECTED" if found else "✗ No pattern"
        if stereo_mode:
            status2 = "✓ DETECTED" if found2 else "✗ Not detected"
            status = f"Left: {status} | Right: {status2}"

        print(f"\r[{captured_count}/{num_images}] {status}  ", end='', flush=True)

        # Wait for user input
        import select
        import sys

        # Non-blocking input check (Unix/Linux only)
        if select.select([sys.stdin], [], [], 0.1)[0]:
            user_input = sys.stdin.readline().strip()

            if user_input.lower() == 'q':
                print("\n\nCapture cancelled by user")
                break

            # ENTER pressed - try to capture
            if found and (not stereo_mode or found2):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                if stereo_mode:
                    # Save both images
                    left_path = f"{output_dir}/left/img_{captured_count:03d}_{timestamp}.jpg"
                    right_path = f"{output_dir}/right/img_{captured_count:03d}_{timestamp}.jpg"
                    cv2.imwrite(left_path, frame)
                    cv2.imwrite(right_path, frame2)
                    print(f"\n  ✓ Saved: {left_path}")
                    print(f"  ✓ Saved: {right_path}")
                else:
                    # Save single image
                    filename = f"{output_dir}/img_{captured_count:03d}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"\n  ✓ Saved: {filename}")

                captured_count += 1
            else:
                print("\n  ✗ Cannot capture - pattern not detected in both cameras")

    cap.release()
    if cap2:
        cap2.release()

    print("\n\n" + "="*60)
    print(f"Capture complete: {captured_count} images saved")
    print(f"Copy '{output_dir}' to your host machine for calibration")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Capture calibration images (headless mode for SSH)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single USB camera
  python capture_calibration.py --camera 0 --output calib_images

  # Single CSI camera (Jetson)
  python capture_calibration.py --use-csi --output calib_images

  # Stereo USB cameras
  python capture_calibration.py --camera 0 --stereo --output calib_stereo

  # Stereo CSI cameras (Jetson)
  python capture_calibration.py --use-csi --stereo --output calib_stereo
        """
    )

    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--output', '-o', type=str, default='calibration_images',
                       help='Output directory (default: calibration_images)')
    parser.add_argument('--pattern', '-p', type=str, default='9x6',
                       help='Checkerboard pattern (default: 9x6 internal corners)')
    parser.add_argument('--count', '-n', type=int, default=30,
                       help='Number of images to capture (default: 30)')
    parser.add_argument('--use-csi', action='store_true',
                       help='Use CSI camera with GStreamer (Jetson)')
    parser.add_argument('--stereo', action='store_true',
                       help='Capture from two cameras for stereo calibration')

    args = parser.parse_args()

    # Parse pattern size
    pattern_parts = args.pattern.lower().split('x')
    if len(pattern_parts) != 2:
        print("ERROR: Pattern must be in format WIDTHxHEIGHT (e.g., 9x6)")
        return

    pattern_size = (int(pattern_parts[0]), int(pattern_parts[1]))

    capture_calibration_images(
        camera_id=args.camera,
        output_dir=args.output,
        pattern_size=pattern_size,
        num_images=args.count,
        use_csi=args.use_csi,
        stereo_mode=args.stereo
    )

if __name__ == "__main__":
    main()

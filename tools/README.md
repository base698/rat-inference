# Tool Layout

Use these canonical paths for utility commands. Root-level compatibility wrappers have been removed.

## Calibration

- `tools/vision/calibration/capture_calibration.py`: capture mono or stereo checkerboard images.
- `tools/vision/calibration/calibrate_camera.py`: solve mono or stereo camera calibration from captured images.

## Hardware

- `tools/hardware/find_motors.py`: scan the Feetech bus for connected motors.
- `tools/hardware/pitch_test.py`: read or write pitch servo positions.
- `tools/hardware/gpio_test.py`: test GPIO/PWM trigger behavior.
- `tools/hardware/servo_test_sysfs.py`: legacy sysfs servo test helper.
- `tools/hardware/trigger_position_test.py`: trigger servo position test helper.

## Dataset And Training

- `tools/vision/dataset/generate-dataset.py`: generate candidate training images.
- `tools/vision/dataset/labeler.py`: label images for YOLO training.
- `tools/vision/dataset/dataset_cleaner.py`: review and clean generated datasets.
- `tools/vision/dataset/extract-frames.sh`: extract frames for dataset building.
- `tools/vision/training/train.py`: train a YOLO model.
- `tools/vision/inference/inference.py`: run image or video inference.

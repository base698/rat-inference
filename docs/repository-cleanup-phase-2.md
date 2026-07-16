# Repository Cleanup Phase 2

Date: 2026-07-16

## Goal

Organize utility scripts into stable locations without changing runtime behavior.

## Changes

- Added the `ratbot/` package as the future home for reusable robot and vision code.
- Moved reusable vision helpers into `ratbot/vision/`.
- Moved calibration tools into `tools/vision/calibration/`.
- Moved hardware diagnostics into `tools/hardware/`.
- Moved dataset and training tools into `tools/vision/dataset/`, `tools/vision/training/`, and `tools/vision/inference/`.
- Kept root-level compatibility wrappers for the previous script names as a temporary migration bridge.

## Canonical Locations

- `ratbot/vision/csi_camera.py`
- `ratbot/vision/yolo_inference.py`
- `tools/vision/calibration/capture_calibration.py`
- `tools/vision/calibration/calibrate_camera.py`
- `tools/hardware/find_motors.py`
- `tools/hardware/pitch_test.py`
- `tools/hardware/gpio_test.py`
- `tools/hardware/servo_test_sysfs.py`
- `tools/hardware/trigger_position_test.py`
- `tools/vision/dataset/labeler.py`
- `tools/vision/dataset/dataset_cleaner.py`
- `tools/vision/dataset/generate-dataset.py`
- `tools/vision/dataset/extract-frames.sh`
- `tools/vision/training/train.py`
- `tools/vision/inference/inference.py`

## Compatibility

At the end of Phase 2, root-level wrappers still existed so the move could be tested without changing command habits all at once.

Phase 6 removed those wrappers after the canonical `tools/hardware/...` and `tools/vision/...` paths were verified. `rt_200.py` now imports shared vision helpers directly from `ratbot/vision`.

## Next Phase

Extract the active tracker into importable robot, vision, workflow, and web modules while keeping `rt_200.py` as a thin CLI wrapper.

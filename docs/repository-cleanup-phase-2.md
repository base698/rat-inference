# Repository Cleanup Phase 2

Date: 2026-07-16

## Goal

Organize utility scripts into stable locations without changing runtime behavior.

## Changes

- Added the `ratbot/` package as the future home for reusable robot and vision code.
- Moved reusable vision helpers into `ratbot/vision/`.
- Moved calibration tools into `tools/calibration/`.
- Moved hardware diagnostics into `tools/hardware/`.
- Moved dataset and training tools into `tools/dataset/`, `tools/training/`, and `tools/inference/`.
- Kept root-level compatibility wrappers for the previous script names.

## Canonical Locations

- `ratbot/vision/csi_camera.py`
- `ratbot/vision/yolo_inference.py`
- `tools/calibration/capture_calibration.py`
- `tools/calibration/calibrate_camera.py`
- `tools/hardware/find_motors.py`
- `tools/hardware/pitch_test.py`
- `tools/hardware/gpio_test.py`
- `tools/hardware/servo_test_sysfs.py`
- `tools/hardware/trigger_position_test.py`
- `tools/dataset/labeler.py`
- `tools/dataset/dataset_cleaner.py`
- `tools/dataset/generate-dataset.py`
- `tools/dataset/extract-frames.sh`
- `tools/training/train.py`
- `tools/inference/inference.py`

## Compatibility

These root-level names remain as wrappers for now:

- `calibrate_camera.py`
- `capture_calibration.py`
- `csi_camera_capture.py`
- `dataset_cleaner.py`
- `extract-frames.sh`
- `find_motors.py`
- `generate-dataset.py`
- `gpio_test.py`
- `inference.py`
- `labeler.py`
- `pitch_test.py`
- `servo_test_sysfs.py`
- `train.py`
- `trigger_position_test.py`
- `yolo_inference.py`

`rt_200.py` is intentionally unchanged in this phase. It still imports the root compatibility modules, which forward to `ratbot/vision`.

## Next Phase

Extract the active tracker into importable robot, vision, workflow, and web modules while keeping `rt_200.py` as a thin CLI wrapper.

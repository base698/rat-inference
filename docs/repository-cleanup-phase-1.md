# Repository Cleanup Phase 1 Inventory

Date: 2026-07-16

Note: this file records the pre-Phase-2 inventory. See `docs/repository-cleanup-phase-2.md` for the current tool layout.

## Recovery Baseline

- Recovery tag: `pre-cleanup-working-main-20260716`
- Baseline commit: `657047f` (`chore: snapshot working stereo calibration`)
- Prior runtime commit: `69a7388` (`fix: stereo depth updates`)

The baseline commit preserves the current working stereo runtime inputs that were not fully captured by the previous commit:

- `config.yaml` with `tracking.crosshair.y_base: 371`
- `STEREO_DEPTH_FIX_SUMMARY.md`
- `calibration_output_recal/stereo_calibration.npz`
- `calibration_output_recal/left/camera_calibration.npz`
- `calibration_output_recal/right/camera_calibration.npz`

To return to the known-good pre-cleanup state:

```bash
git checkout pre-cleanup-working-main-20260716
```

Or create a branch from it:

```bash
git switch -c restore/pre-cleanup-working-main pre-cleanup-working-main-20260716
```

## Current Working Runtime

The Jetson stereo tracker is still rooted at `rt_200.py`.

Current manual test command:

```bash
uv run --extra jetson python rt_200.py \
  --enable-camera \
  --use-csi \
  --disable-detection \
  --stereo \
  --calibration calibration_output_recal/stereo_calibration.npz \
  --baseline-override 57.5 \
  --port /dev/ttyACM0
```

The older Raspberry Pi trap script is archived as legacy code:

- `archive/raspberry-pi/rt_100.py`

## Inventory

Runtime and robot control:

- `rt_200.py`: active Jetson stereo tracker, web UI, depth overlay, servo control.
- `archive/raspberry-pi/rt_100.py`: legacy Raspberry Pi trap/detection/trigger script.
- `config.yaml`: active runtime calibration, servo, crosshair, and compensation config.

Camera and calibration utilities:

- `calibrate_camera.py`: mono/stereo chessboard calibration.
- `capture_calibration.py`: capture helper with web workflow and detection gating.
- `csi_camera_capture.py`: CSI camera helper.

Robot hardware utilities:

- `find_motors.py`: Feetech bus scan.
- `pitch_test.py`: pitch servo alignment checks.
- `gpio_test.py`: GPIO trigger checks.
- `servo_test_sysfs.py`: legacy/sysfs servo test.
- `trigger_position_test.py`: trigger servo position checks.

Training and dataset tools:

- `labeler.py`: Tkinter dataset labeler.
- `train.py`: YOLO training CLI.
- `dataset_cleaner.py`: dataset cleanup helper.
- `generate-dataset.py`: dataset generation helper.
- `extract-frames.sh`: frame extraction helper.

Detection/inference scripts:

- `yolo_inference.py`: reusable YOLO inference helper candidate.
- `inference.py`: older inference entry point candidate for review.
- `main.py`: likely older Roboflow/demo-style entry point candidate for archive/removal review.

Scratch or review-before-removal files:

- `test.py`
- `test2.py`
- `test.sh`
- loose images such as `bus.jpg`, `test_frame.jpg`, `test_capture_*.jpg`
- timestamped `.bak-*` files and runtime logs

## Phase 2 Direction

Prefer a top-level `ratbot/` package first so direct Jetson commands like `uv run python rt_200.py` keep working without packaging or `PYTHONPATH` changes. Keep root-level compatibility wrappers while code moves into modules.

Proposed module targets:

- `ratbot/vision/`: camera sources, CSI camera capture, stereo depth, calibration loading, YOLO inference.
- `ratbot/robot/`: pan/tilt interfaces, Feetech implementation, trigger interfaces, GPIO trigger, simulation, geometry.
- `ratbot/web/`: FastAPI app factory, API schemas, static UI.
- `ratbot/workflows/`: tracker orchestration and trap workflow.
- `tools/calibration/`: calibration capture and calibration CLIs.
- `tools/hardware/`: motor, GPIO, pitch, and trigger diagnostic CLIs.
- `tools/dataset/`: labeler, cleaner, frame extraction, dataset generation.

The reusable controller shape should let the web interface run against any robot implementation that provides:

- a camera or stereo camera source
- optional depth lookup for an image point
- a two-axis pan/tilt actuator
- optional trigger actuator

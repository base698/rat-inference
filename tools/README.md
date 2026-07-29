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

## RealSense D435

Two views of the same explorer — device introspection, color/depth/IR streaming,
depth-to-color alignment, post-processing filters, IR emitter and visual preset
control, depth probing, and point cloud export.

- `tools/vision/realsense_explorer.py`: desktop OpenCV window. macOS needs root
  to claim the USB device: `sudo .venv/bin/python tools/vision/realsense_explorer.py`.
- `tools/vision/realsense_web.py`: same thing served over HTTP, for the headless
  Jetson. Defaults to port 8090, leaving 8000 free for `rt_200.py`.

On the Jetson these run under the system `python3` (3.10), not the project venv:
Intel ships aarch64 `pyrealsense2` wheels for cp39/310/312 but not cp311.

```bash
python3 -m pip install --user pyrealsense2      # one time
cd ~/rat-inference
setsid python3 -u tools/vision/realsense_web.py --port 8090 \
  </dev/null >run_logs/realsense_web.log 2>&1 &
```

Then open `http://<jetson-ip>:8090/`. Add `--info` to dump device capabilities
and factory calibration as JSON without starting the server. To stop it, use
`pkill -f "[r]ealsense_web.py"` — the bracket keeps the pattern from matching
the shell running your own command.

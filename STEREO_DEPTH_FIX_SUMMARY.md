# Stereo Depth Tracking Fix Summary

Date: 2026-07-15

This file summarizes the work done to restore camera video, servo control, stereo depth, and laser/crosshair alignment in `~/rat-inference`.

## Current Working Run Command

```bash
cd ~/rat-inference
uv run --extra jetson python rt_200.py \
  --enable-camera \
  --use-csi \
  --disable-detection \
  --stereo \
  --calibration tools/vision/calibration/output_recal/stereo_calibration.npz \
  --baseline-override 57.5 \
  --port /dev/ttyACM0
```

The web UI is available at:

```text
http://<jetson-ip>:8000
```

## Camera Video Fixes

- The original run showed `Camera: DISABLED` because `rt_200.py` requires `--enable-camera`.
- The Jetson CSI cameras work through `nvarguscamerasrc`, so the app must also use `--use-csi`.
- Added `--video-only` convenience mode for quick CSI camera streaming with no servos and no detection:

```bash
uv run python rt_200.py --video-only
```

- Verified `/stream-frame` returns a valid `640x480` JPEG.

## Servo Fixes

- Tracking servos are enabled by default; there is no `--enable-servos` flag.
- The Feetech dependency must be run through the Jetson optional group:

```bash
uv run --extra jetson python rt_200.py ...
```

- `find_motors.py` confirmed the expected motor IDs:

```text
ID 1: yaw
ID 5: pitch
```

- The user needed access to `/dev/ttyACM0`; adding `base698` to `dialout` fixed device access.
- Fixed a stale `SIGALRM` timeout in `connect_servos()` that could crash the server later in the main loop.
- Added `motor_lock` to serialize Feetech reads and writes, fixing `Port is in use!` collisions between camera-thread position reads and UI click writes.

## Stereo Calibration and Depth Fixes

The old calibration had:

```text
Stereo RMS: 3.826px
Baseline: 135.31mm
```

It caused severe rectification distortion and mapped normal image points far outside the `640x480` rectified frame.

Recalibration was done using a `6x4` internal-corner checkerboard displayed on a screen:

```bash
uv run --extra jetson python tools/vision/calibration/capture_calibration.py \
  --web \
  --use-csi \
  --stereo \
  --pattern 6x4 \
  --output tools/vision/calibration/images_recal \
  --port 8010
```

The capture helper only saves stereo pairs when the checkerboard is detected in both cameras.

Calibration command:

```bash
uv run --extra jetson python tools/vision/calibration/calibrate_camera.py \
  --stereo \
  --left "tools/vision/calibration/images_recal/left/*.jpg" \
  --right "tools/vision/calibration/images_recal/right/*.jpg" \
  --pattern 6x4 \
  --square-size 21 \
  --output tools/vision/calibration/output_recal
```

New calibration results:

```text
Left RMS: 0.065px
Right RMS: 0.046px
Stereo RMS: 1.486px
Solved baseline: 42.61mm
Measured physical baseline: 57.5mm
```

The physical lens-center distance is `57.5mm`, so runtime should use:

```bash
--baseline-override 57.5
```

## Stereo Runtime Code Fixes

- Added stereo rectification maps using `cv2.stereoRectify()` and `cv2.initUndistortRectifyMap()`.
- Kept the displayed web image unrectified; rectification is now only used internally for depth math.
- Added depth diagnostics on the overlay and console, for example:

```text
-- no valid disparity near (x, y)
-- rectified point out of frame (x, y)
DepthYAdjust: 12.5px
```

- Changed rectification from hard crop (`alpha=0`) to OpenCV default (`alpha=-1`) to avoid throwing points off-screen.
- Added a common rectified image shift so the raw image center maps near the rectified center. This preserves disparity while moving the working point away from invalid border regions.
- Added automatic negative-disparity support when the rectification matrix indicates that camera order/sign expects negative disparities.
- Expanded disparity search to:

```text
minDisparity = -192
numDisparities = 256
sign = -1
```

for the current calibration.

## Crosshair and Laser Alignment

The previous pitch-only Y compensation was a fallback table in `config.yaml`.

Current pitch fallback points:

```yaml
pitch_compensation:
  enabled: true
  pitch_min: 100
  pitch_max: 550
  y_offset_at_min: -33.8
  y_offset_at_max: -30.0
  points:
    - pitch: 100
      offset: -33.8
    - pitch: 240
      offset: -18.9
    - pitch: 250
      offset: -17.8
    - pitch: 415
      offset: -21.2
    - pitch: 550
      offset: -30.0
```

A better model was added using stereo depth and the measured camera-to-laser vertical offset.

The laser exits below the camera center by about `55mm`, so `config.yaml` now includes:

```yaml
depth_crosshair_compensation:
  enabled: true
  laser_vertical_offset_mm: 55.0
  reference_distance_mm: 1000.0
  max_adjust_px: 80.0
```

The depth correction uses:

```text
depth_adjust_px = fy * laser_vertical_offset_mm * (1/depth_mm - 1/reference_distance_mm)
```

With the current configuration, expected Y adjustments are approximately:

```text
500mm  -> +37.6px
750mm  -> +12.5px
1000mm -> 0.0px
1500mm -> -12.5px
2000mm -> -18.8px
```

Positive adjustment moves the crosshair downward in the image.

## Useful Test Commands

Camera-only CSI stream:

```bash
uv run python rt_200.py --video-only
```

Stereo camera-only test:

```bash
uv run --extra jetson python rt_200.py \
  --enable-camera \
  --use-csi \
  --disable-detection \
  --disable-servos \
  --no-connect \
  --stereo \
  --calibration tools/vision/calibration/output_recal/stereo_calibration.npz \
  --baseline-override 57.5
```

Servo discovery:

```bash
uv run --extra jetson python find_motors.py
```

## Important Backups Created

Several timestamped backups were created during the repair, including:

```text
rt_200.py.bak-20260715-videoonly
rt_200.py.bak-20260715-stereo-rectify
rt_200.py.bak-20260715-stereo-display-fix
rt_200.py.bak-20260715-depth-diagnostics
rt_200.py.bak-20260715-negative-disparity
rt_200.py.bak-20260715-rectified-shift
rt_200.py.bak-20260715-servo-lock
rt_200.py.bak-20260715-pitch-points
rt_200.py.bak-20260715-depth-crosshair
config.yaml.bak-20260715-pitch-y-offset
config.yaml.bak-20260715-pitch-min-small-tweak
config.yaml.bak-20260715-pitch-points
config.yaml.bak-20260715-depth-crosshair
```

## Repository Cleanup Status

Phase 1 tagged the known-good starting point as:

```text
pre-cleanup-working-main-20260716
```

Phase 2 organized the utility scripts into stable locations under `tools/` and moved reusable vision helpers into `ratbot/vision/`, while temporarily leaving root-level compatibility wrappers in place.

The bad original calibration artifact set has now been removed:

```text
calibration_output/
calibration_images/
```

The current source-of-truth calibration set is tracked in git:

```text
tools/vision/calibration/images_recal/
tools/vision/calibration/output_recal/
```

`rt_200.py` now auto-selects `tools/vision/calibration/output_recal/stereo_calibration.npz` when stereo mode is enabled and no explicit calibration path is provided. Runtime should still pass `--baseline-override 57.5` because the measured physical lens-center baseline is more trustworthy than the solved baseline from the screen-based calibration.

Phase 3 is the modularization pass: keep `rt_200.py` usable as the CLI entry point, but extract the robot/camera/stereo/web pieces into importable modules so the controller can be reused with another stereo camera and 2-axis servo robot.

The first Phase 3 slice extracts the FastAPI controller into:

```text
ratbot/web/control_api.py
```

`rt_200.py` still owns CLI parsing and the current `CameraTracker`, but it now passes the live tracker into the web controller with `control_api.set_tracker(...)`. This makes the web layer depend on a tracker-like object instead of global route functions living inside the robot runtime script.

The Phase 4 slice adds the explicit robot protocol:

```text
ratbot/robot/interfaces.py
```

`TrackerRobot` documents the status attributes, frame access, detection data, movement commands, and trigger action that the web controller needs. `ratbot/web/control_api.py` now depends on this protocol instead of `Any`, so another stereo camera plus 2-axis servo robot can reuse the web controller by implementing the same surface.

The Phase 5 slice extracts crosshair and laser-depth aiming behavior into:

```text
ratbot/robot/aiming.py
```

`CrosshairAiming` now owns yaw-to-X compensation, pitch-to-Y compensation, pitch point interpolation, and stereo-depth laser Y adjustment. `rt_200.py` still exposes the existing `get_target_crosshair_x(...)` and `get_target_crosshair_y(...)` wrappers, so the web controller and tracker call sites remain stable while the robot package gains real reusable behavior.

The follow-up cleanup consolidates all vision utility code and artifacts under:

```text
tools/vision/
```

That includes calibration scripts and outputs, dataset tools, training, inference, sample prompts, sample assets, and the legacy `main.py` image experiment. Root-level utility wrappers were removed after copying the old root layout to:

```text
/tmp/rat-inference-root-cleanup-20260716-125144
```

## Remaining Notes

- The current stereo depth seems plausible but the stereo RMS is still `1.486px`, so a rigid printed calibration target should improve accuracy.
- Screen-based calibration works for debugging, but a printed or mounted board is better because it avoids screen glare, focus artifacts, and pose flex.
- If depth scale is wrong, first verify `--baseline-override 57.5` is present.
- If vertical laser alignment is wrong only at certain distances, tune `laser_vertical_offset_mm` or `reference_distance_mm` before changing pitch points.

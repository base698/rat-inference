# Repository Cleanup Phase 3

Date: 2026-07-16

## Goal

Start splitting the active tracker into reusable modules while preserving the existing `rt_200.py` command-line behavior.

## First Slice

- Added `ratbot/web/control_api.py` as the FastAPI web controller.
- Added `ratbot/web/__init__.py` exports for the web controller package.
- Kept `rt_200.py` as the executable CLI entry point.
- Left `CameraTracker` in `rt_200.py` for this slice.
- Passed the live tracker into the web controller through `control_api.set_tracker(...)`.

## Boundary

The web layer now depends on a tracker-like object with the methods and attributes already used by the UI:

- `connected`
- `current_yaw`
- `current_pitch`
- `camera_active`
- `detection_count`
- `trigger_servo_enabled`
- `get_detection_data()`
- `get_latest_frame_bytes()`
- `set_yaw(...)`
- `set_pitch(...)`
- `trigger_action_servo()`
- `move_to_pixel(...)`

That is the first explicit interface between the web controller and the robot runtime. A later pass can formalize this as a protocol and move `CameraTracker` itself into `ratbot/robot/`.

## Next Slice

Extract robot-side pieces from `CameraTracker` into importable modules:

- servo bus connection and locking
- camera capture lifecycle
- stereo depth computation
- detection loop
- tracker orchestration

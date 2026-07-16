# Repository Cleanup Phase 4

Date: 2026-07-16

## Goal

Make the web-to-robot boundary explicit so the web controller can be reused with another stereo camera and 2-axis servo robot.

## Changes

- Added `ratbot/robot/interfaces.py`.
- Added `ratbot/robot/__init__.py`.
- Introduced `TrackerRobot`, a protocol describing the robot surface the web controller needs.
- Updated `ratbot/web/control_api.py` to depend on `TrackerRobot` instead of `Any`.

## TrackerRobot Surface

The web controller expects:

- status attributes: `connected`, `camera_active`, `detection_count`
- axis state: `current_yaw`, `current_pitch`
- optional trigger state: `trigger_servo_enabled`
- camera frame access: `get_latest_frame_bytes()`
- detection state access: `get_detection_data()`
- manual movement: `set_yaw(...)`, `set_pitch(...)`, `move_to_pixel(...)`
- optional trigger action: `trigger_action_servo()`

`rt_200.CameraTracker` already satisfies this protocol without inheritance, so this phase should not change runtime behavior.

## Next Phase

Move the active robot implementation out of `rt_200.py` in small pieces:

- extract servo configuration and bus access
- extract camera startup/shutdown lifecycle
- extract stereo calibration/depth helpers
- keep `rt_200.py` as the CLI composition layer

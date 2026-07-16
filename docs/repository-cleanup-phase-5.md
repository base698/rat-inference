# Repository Cleanup Phase 5

Date: 2026-07-16

## Goal

Move a real piece of robot behavior out of `rt_200.py` while keeping runtime behavior stable.

## Changes

- Added `ratbot/robot/aiming.py`.
- Added dataclasses for yaw, pitch, and depth crosshair compensation.
- Moved target crosshair X/Y math into `CrosshairAiming`.
- Moved depth-based laser Y adjustment into `CrosshairAiming.depth_adjust_px(...)`.
- Kept `get_target_crosshair_x(...)` and `get_target_crosshair_y(...)` wrappers in `rt_200.py` so the web controller and tracker call sites stay stable.

## Boundary

`rt_200.py` still owns YAML loading and CLI/runtime composition. `ratbot.robot.aiming` now owns the reusable aiming model:

- yaw-position-to-crosshair-X compensation
- pitch-position-to-crosshair-Y compensation
- optional pitch point interpolation
- stereo-depth laser vertical offset compensation

This is the first robot behavior extraction after the web/controller split.

## Next Phase

The next safe extraction candidates are:

- servo axis config and clamp/write helpers
- stereo calibration/depth computation
- camera lifecycle setup

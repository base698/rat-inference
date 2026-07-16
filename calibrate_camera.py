#!/usr/bin/env python3
"""Compatibility wrapper for tools/calibration/calibrate_camera.py."""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "tools" / "calibration" / "calibrate_camera.py"),
    run_name="__main__",
)

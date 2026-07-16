#!/usr/bin/env python3
"""Compatibility wrapper for tools/calibration/capture_calibration.py."""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "tools" / "calibration" / "capture_calibration.py"),
    run_name="__main__",
)

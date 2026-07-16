#!/usr/bin/env python3
"""Compatibility wrapper for tools/hardware/find_motors.py."""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "tools" / "hardware" / "find_motors.py"),
    run_name="__main__",
)

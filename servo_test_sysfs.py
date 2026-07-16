#!/usr/bin/env python3
"""Compatibility wrapper for tools/hardware/servo_test_sysfs.py."""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "tools" / "hardware" / "servo_test_sysfs.py"),
    run_name="__main__",
)

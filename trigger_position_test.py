#!/usr/bin/env python3
"""Compatibility wrapper for tools/hardware/trigger_position_test.py."""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "tools" / "hardware" / "trigger_position_test.py"),
    run_name="__main__",
)

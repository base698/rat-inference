#!/usr/bin/env python3
"""Compatibility wrapper for tools/training/train.py."""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "tools" / "training" / "train.py"),
    run_name="__main__",
)

#!/usr/bin/env python3
"""Compatibility wrapper for tools/dataset/labeler.py."""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "tools" / "dataset" / "labeler.py"),
    run_name="__main__",
)

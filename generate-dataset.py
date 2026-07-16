#!/usr/bin/env python3
"""Compatibility wrapper for tools/dataset/generate-dataset.py."""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "tools" / "dataset" / "generate-dataset.py"),
    run_name="__main__",
)

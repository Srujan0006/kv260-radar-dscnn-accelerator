#!/usr/bin/env python3
"""Readable entry point for record-wise spectrogram cache generation."""

from pathlib import Path
import runpy

SCRIPT = Path(__file__).with_name("01_make_recordwise_windows.py")
runpy.run_path(str(SCRIPT), run_name="__main__")

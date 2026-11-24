#!/usr/bin/env python3
"""Repo-standard entry point for the Vitis AI quantization script."""

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts/quantize_vai_pytorch.py"), run_name="__main__")

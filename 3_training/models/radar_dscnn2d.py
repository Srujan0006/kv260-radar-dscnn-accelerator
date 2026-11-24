"""Depthwise-separable radar CNN used by the record-wise training pipeline.

The implementation lives in the installable ``radar2dcnn`` package so export,
quantization, evaluation, and training all instantiate exactly the same graph.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from radar2dcnn.model import DWConvBlock, RadarDSCNN2D

__all__ = ["DWConvBlock", "RadarDSCNN2D"]

"""Prediction head exports used by TiSAM decoders and segmentors.

The decoder stack imports `ClassPredictor` and `MaskPredictor` from this package
when constructing mask-class logits for downstream training and inference.
"""

from __future__ import annotations

from .class_predictor import ClassPredictor
from .mask_predictor import MaskPredictor

__all__ = ["ClassPredictor", "MaskPredictor"]

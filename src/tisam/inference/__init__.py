"""Image and WSI inference; install tisam[inference] for file-based workflows."""

from .evaluate import evaluate
from .predict import predict
from .wsi import segment_wsi

__all__ = ["predict", "segment_wsi", "evaluate"]

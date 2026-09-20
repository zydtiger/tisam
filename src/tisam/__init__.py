"""Installable dual-encoder TiSAM architecture; optional workflows import separately."""

from .checkpointing.weights import load_model
from .config.model import ModelConfig
from .model.segmentor import TiSAM

__version__ = "0.1.0"
__all__ = ["TiSAM", "ModelConfig", "load_model", "__version__"]

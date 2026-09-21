"""Training API loaded on demand; losses also support inference-only environments."""

import importlib
import sys
from typing import Any

__all__ = ["train"]


def __getattr__(name: str) -> Any:
    """Load the Mammoth-backed trainer only when the public training API is requested."""
    if name != "train":
        raise AttributeError(name)
    if sys.version_info < (3, 12):
        raise ImportError("TiSAM training requires Python >=3.12 and tisam[train].")
    return importlib.import_module(".runner", __name__).train

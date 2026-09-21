"""Training API loaded on demand; losses also support inference-only environments."""

import importlib
from typing import Any

__all__ = ["train"]


def __getattr__(name: str) -> Any:
    """Load the Mammoth-backed trainer only when the public training API is requested."""
    if name != "train":
        raise AttributeError(name)
    return importlib.import_module(".runner", __name__).train

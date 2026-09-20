"""
Class predictor component for SAM3 Mask2Former decoder.

Predicts class labels for each object query using a simple linear projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from prettyterm import get_logger

logger = get_logger(__name__)


class ClassPredictor(nn.Module):
    """
    Class predictor for object queries.

    Uses a simple linear projection to predict class logits for each query.
    Each query predicts scores for all classes (softmax applied during loss).

    Architecture:
        obj_queries (B, N, d) -> Linear -> class_logits (B, N, num_classes)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 11,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes  # Number of semantic classes (does not include void)
        self.total_classes = num_classes + 1  # +1 for void/background class

        # Linear projection to class logits (includes void class)
        self.class_embed = nn.Linear(d_model, self.total_classes)

        logger.success(
            f"ClassPredictor initialized: d_model={d_model}, "
            f"num_classes={num_classes}, total_classes={self.total_classes}"
        )

    def forward(self, obj_queries: torch.Tensor) -> torch.Tensor:
        """
        Predict class logits for object queries.

        Args:
            obj_queries: (B, num_queries, d_model) refined object queries

        Returns:
            class_logits: (B, num_queries, total_classes) class logits (includes void)
        """
        # Linear projection: (B, N, d_model) -> (B, N, total_classes)
        class_logits = self.class_embed(obj_queries)

        return class_logits

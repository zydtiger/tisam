"""
Mask predictor component for SAM3 Mask2Former decoder.

Uses dot-product similarity between object queries and pixel embeddings
to generate mask predictions (following SAM3's MaskPredictor pattern).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from prettyterm import get_logger

from ..layers import MLP

logger = get_logger(__name__)


class MaskPredictor(nn.Module):
    """
    Mask predictor using dot-product similarity.

    Projects mask embeddings from transformer decoder and computes dot product
    with pixel embeddings to generate mask logits.

    Architecture:
        mask_embeds (N, B, d_model)
            ↓
        permute to (B, N, d_model)
            ↓
        MLP (projection) -> mask_proj
            ↓
        mask_proj (B, N, d_model)
            ↓
        dot product with pixel_embed (B, d_model, H, W)
            ↓
        mask_preds (B, N, H, W)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        # MLP to project mask embeddings to mask dimension
        self.mask_proj = MLP(d_model, d_model, d_model, num_layers)

        logger.success(f"MaskPredictor initialized: d_model={d_model}, num_layers={num_layers}")

    def forward(
        self,
        mask_embeds: torch.Tensor,
        pixel_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate mask predictions via dot-product similarity.

        Args:
            mask_embeds: (num_queries, B, transformer_dim) mask embeddings from transformer decoder
            pixel_embed: (B, mask_dim, H, W) pixel embeddings from PixelDecoder

        Returns:
            mask_preds: (B, num_queries, H, W) mask logits
        """
        # Permute from (N, B, D) to (B, N, D) for MLP processing
        # (num_queries, B, transformer_dim) -> (B, num_queries, transformer_dim)
        mask_embeds = mask_embeds.permute(1, 0, 2)

        # Project mask embeddings to mask dimension
        # (B, N, transformer_dim) -> (B, N, mask_dim)
        mask_proj = self.mask_proj(mask_embeds)

        # Dot product: compute similarity between each query and each pixel.
        # Keep this as an explicit batched matmul so torch.compile sees simple
        # 3D tensor contractions instead of an einsum expression.
        b, _c, h, w = pixel_embed.shape
        q = mask_proj.shape[1]
        pixel_flat = pixel_embed.flatten(2)
        mask_preds = torch.bmm(mask_proj, pixel_flat).reshape(b, q, h, w)

        return mask_preds

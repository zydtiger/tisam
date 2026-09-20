"""
Complete Mask2Former Decoder for semantic segmentation.

Orchestrates all Mask2Former components:
- PixelDecoder for multi-scale feature fusion
- TransformerDecoder with masked cross-attention
- MaskPredictor for mask generation
- ClassPredictor for classification

This replaces the old sam3_mask2former_decoder.py with true Mask2Former architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from prettyterm import get_logger

from ..encoders.extra_encoder import ExtraFeatures
from ..feature_pyramid import SAM3_FEATURE_PYRAMID_SPEC, FeaturePyramidSpec
from ..predictors import ClassPredictor, MaskPredictor
from .pixel_decoder import PixelDecoder
from .transformer_decoder import Mask2FormerTransformerDecoder

logger = get_logger(__name__)


class Mask2FormerDecoder(nn.Module):
    """
    Complete Mask2Former decoder for semantic segmentation.

    Combines:
    1. PixelDecoder for multi-scale FPN fusion
    2. TransformerDecoder with masked cross-attention
    3. MaskPredictor for mask generation via dot product
    4. ClassPredictor for classification

    Key features:
    - Masked cross-attention: masks from previous layer gate attention
    - Multi-scale cycling: each layer attends to different feature scale
    - Deep supervision: all intermediate mask predictions are output
    - Semantic aggregation: multi-instance predictions aggregated to semantic

    Args:
        num_classes: Number of semantic classes (excluding void/background)
        total_queries: Total number of learnable queries
        d_model: Transformer dimension
        num_layers: Number of decoder layers
        n_heads: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout probability
        upsampling_stages: Number of upsampling stages in PixelDecoder
        output_hw: Optional output resolution for mask logits (H, W)
        use_sdpa_attn: Whether transformer cross-attention uses PyTorch SDPA.
    """

    def __init__(
        self,
        num_classes: int = 11,
        total_queries: int = 36,
        d_model: int = 256,
        num_layers: int = 6,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        upsampling_stages: int = 3,
        extra_embed_dim: int | None = 1280,
        use_cross_attn: bool = False,
        hard_attn: bool = False,
        use_sdpa_attn: bool = True,
        output_hw: tuple[int, int] | None = None,
        sam_proj: bool = False,
        early_interpolation: bool = True,
        use_pos_embed: bool = True,
        pos_embed_mode: str | None = None,
        learned_pos_embed_init: str = "normal",
        learned_pos_embed_std: float = 0.02,
        learned_image_pos_scope: str = "per_level",
        learned_query_pos: bool = True,
        learned_image_pos: bool = True,
        feature_pyramid_spec: FeaturePyramidSpec = SAM3_FEATURE_PYRAMID_SPEC,
    ):
        super().__init__()
        self.num_classes = num_classes  # Semantic classes only
        self.total_classes = num_classes + 1  # +1 for void/background
        self.total_queries = total_queries
        self.num_queries = total_queries
        self.d_model = d_model
        self.num_layers = num_layers
        self.feature_pyramid_spec = feature_pyramid_spec

        # Create mask_predictor FIRST (will be shared with transformer_decoder)
        self.mask_predictor = MaskPredictor(d_model=d_model, num_layers=3)

        # Pixel decoder for multi-scale FPN fusion
        self.pixel_decoder = PixelDecoder(
            d_model=d_model,
            fpn_channels=list(feature_pyramid_spec.channels),
            upsampling_stages=upsampling_stages,
            extra_embed_dim=extra_embed_dim,  # Pass for projection layer
            use_cross_attn=use_cross_attn,
            output_hw=output_hw,
            sam_proj=sam_proj,
            early_interpolation=early_interpolation,
            feature_pyramid_spec=feature_pyramid_spec,
        )

        # Transformer decoder with masked cross-attention
        # PASS the shared mask_predictor to avoid duplicate mask prediction logic
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            d_model=d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_queries=total_queries,
            num_feature_levels=len(feature_pyramid_spec.spatial_shapes),
            mask_predictor=self.mask_predictor,  # Shared mask predictor
            hard_attn=hard_attn,
            use_sdpa_attn=use_sdpa_attn,
            use_pos_embed=use_pos_embed,
            pos_embed_mode=pos_embed_mode,
            learned_pos_embed_init=learned_pos_embed_init,
            learned_pos_embed_std=learned_pos_embed_std,
            learned_image_pos_scope=learned_image_pos_scope,
            learned_query_pos=learned_query_pos,
            learned_image_pos=learned_image_pos,
            feature_shapes=feature_pyramid_spec.spatial_shapes,
        )

        # Class predictor (linear projection)
        self.class_predictor = ClassPredictor(
            d_model=d_model,
            num_classes=num_classes,
        )

        logger.success(
            f"Mask2FormerDecoder initialized: "
            f"num_classes={num_classes}, total_classes={self.total_classes}, "
            f"total_queries={self.total_queries}, d_model={d_model}, "
            f"num_layers={num_layers}, shared_mask_predictor=True"
        )

    def prepare_compile_metadata(self) -> None:
        """Verify fixed pixel and positional metadata before compiled tracing."""
        device = next(self.pixel_decoder.parameters()).device
        if self.pixel_decoder.spatial_shapes.device != device:
            raise RuntimeError(
                "Pixel-decoder spatial metadata must follow the decoder device before compilation."
            )
        if self.pixel_decoder.level_start_index.device != device:
            raise RuntimeError(
                "Pixel-decoder level metadata must follow the decoder device before compilation."
            )
        self.transformer_decoder.prepare_compile_metadata()

    def forward(
        self,
        backbone_fpn: list[torch.Tensor],
        extra_features: ExtraFeatures | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through Mask2Former decoder.

        Args:
            backbone_fpn: List of FPN features at multiple scales
                ordered from highest resolution (P2) to lowest (P5), with
                channels and fixed spatial shapes defined by
                `feature_pyramid_spec`.
            extra_features: Optional final histopathology map or P3/P4/P5-ordered
                deep feature tuple from `ExtraEncoder`.

        Returns:
            dict with:
                - pred_masks: (num_layers, b, n, h, w) intermediate mask predictions
                  at mask_features resolution (output_hw if configured)
                - pred_logits: (num_layers, b, n, total_classes) class predictions
                - final_masks: (b, n, h, w) final mask predictions
                - final_logits: (b, n, total_classes) final class predictions
                - semantic_masks: (b, total_classes, h, w) aggregated semantic masks
        """
        # === Step 1: Pixel decoding ===
        pixel_out = self.pixel_decoder(backbone_fpn, extra_features)
        mask_features = pixel_out["mask_features"]  # (b, d_model, h, w)
        multi_scale_features = pixel_out["multi_scale_features"]  # list of (b, c, h, w)

        # === Step 2: Transformer decoding with iterative mask refinement ===
        multi_scale_pos = self.transformer_decoder.image_positions(multi_scale_features)
        hs = self.transformer_decoder(
            multi_scale_features=multi_scale_features,
            multi_scale_pos=multi_scale_pos,
        )

        # === Step 3: Predict masks at all layers (deep supervision) ===
        # Transpose for mask predictor: (num_layers, n, b, d_model) -> (num_layers, b, n, d_model)
        hs_batch_first = hs.permute(0, 2, 1, 3)  # (num_layers, b, n, d_model)

        # Predict masks at all layers
        # We need to handle each layer separately since mask_features has fixed resolution
        # but the mask_embeds were predicted at different scales
        pred_masks = self._predict_all_layer_masks(
            hs_batch_first,  # (num_layers, b, n, d_model)
            mask_features,  # (b, mask_dim, h, w)
        )  # (num_layers, b, n, h, w) at high resolution

        # === Step 4: Predict class logits at all layers ===
        # hs: (num_layers, n, b, d_model) -> permute to (num_layers, b, n, d_model)
        # then apply class predictor to each layer
        pred_logits = self._predict_all_layer_logits(
            hs_batch_first,  # (num_layers, b, n, d_model)
        )  # (num_layers, b, n, total_classes)

        # === Step 5: Final predictions ===
        final_masks = pred_masks[-1]  # (b, n, h, w)
        final_logits = pred_logits[-1]  # (b, n, total_classes)

        # === Step 6: Aggregate multi-instance to semantic ===
        semantic_masks = self._aggregate_to_semantic(final_masks, final_logits)
        # (b, total_classes, h, w)

        return {
            "pred_masks": pred_masks,  # (num_layers, b, n, h, w)
            "pred_logits": pred_logits,  # (num_layers, b, n, total_classes)
            "final_masks": final_masks,  # (b, n, h, w)
            "final_logits": final_logits,  # (b, n, total_classes)
            "semantic_masks": semantic_masks,  # (b, total_classes, h, w)
        }

    def _predict_all_layer_masks(
        self,
        hs: torch.Tensor,  # (num_layers, b, n, d_model)
        mask_features: torch.Tensor,  # (b, mask_dim, h, w)
    ) -> torch.Tensor:
        """
        Predict masks at all decoder layers.

        Args:
            hs: (num_layers, b, n, d_model) query embeddings at each layer
            mask_features: (b, mask_dim, h, w) high-res features for mask prediction

        Returns:
            pred_masks: (num_layers, b, n, h, w) masks at high resolution
        """
        num_layers, b, n, _ = hs.shape
        mask_h, mask_w = mask_features.shape[-2], mask_features.shape[-1]
        pred_masks = mask_features.new_empty((num_layers, b, n, mask_h, mask_w))

        for layer_idx in range(num_layers):
            layer_hs = hs[layer_idx]  # (b, n, d_model)

            # MaskPredictor expects (n, b, d), so permute first
            layer_hs_permuted = layer_hs.permute(1, 0, 2)  # (b, n, d) -> (n, b, d)

            # Predict masks at high resolution
            layer_masks = self.mask_predictor(layer_hs_permuted, mask_features)
            # (b, n, h_high, w_high)

            pred_masks[layer_idx] = layer_masks

        return pred_masks

    def _predict_all_layer_logits(
        self,
        hs: torch.Tensor,  # (num_layers, b, n, d_model)
    ) -> torch.Tensor:
        """
        Predict class logits at all decoder layers.

        Args:
            hs: (num_layers, b, n, d_model) query embeddings at each layer

        Returns:
            pred_logits: (num_layers, b, n, total_classes) class logits
        """
        num_layers, b, n, d = hs.shape

        # Reshape to apply class predictor to all layers at once
        # (num_layers, b, n, d_model) -> (num_layers * b, n, d_model)
        hs_flat = hs.reshape(num_layers * b, n, d)

        # Apply class predictor
        logits_flat = self.class_predictor(hs_flat)
        # (num_layers * b, n, total_classes)

        # Reshape back
        pred_logits = logits_flat.reshape(num_layers, b, n, self.total_classes)

        return pred_logits

    def _aggregate_to_semantic(
        self,
        pred_masks: torch.Tensor,  # (b, n, h, w)
        pred_logits: torch.Tensor,  # (b, n, total_classes)
    ) -> torch.Tensor:
        """
        Aggregate multi-instance detections to semantic segmentation using soft attention.

        SOFT ATTENTION APPROACH:
        - Every query contributes to every class based on its learned affinity
        - No hard assignment or query-to-class mapping required
        - Gradients flow smoothly through all query-class pairs
        - Classes cannot "disappear" since all queries always contribute

        Mathematical formulation:
            attention_weights[b, n, c] = softmax(pred_logits[b, n, :])[c]
            semantic_masks[b, c, h, w] = sum_n (attention_weights[b, n, c] * pred_masks[b, n, h, w])

        Args:
            pred_masks: (b, n, h, w) mask predictions for each query
            pred_logits: (b, n, total_classes) class logits (includes void)

        Returns:
            semantic_masks: (b, total_classes, h, w) semantic segmentation masks
        """
        # === Compute attention weights ===
        # attention_weights[b, n, c] = how much query n contributes to class c
        # Shape: (b, n, c)
        # Use log_softmax for numerical stability, then exp to get probabilities
        log_attention_weights = torch.nn.functional.log_softmax(pred_logits, dim=-1)
        attention_weights = torch.exp(log_attention_weights)

        # === Aggregate using soft attention ===
        # semantic_masks[b, c, h, w] = sum_n attention_weights[b, n, c] * pred_masks[b, n, h, w]
        b, _n, h, w = pred_masks.shape
        c = attention_weights.shape[-1]
        pred_masks_flat = pred_masks.flatten(2)
        semantic_masks_flat = torch.bmm(
            attention_weights.transpose(1, 2),
            pred_masks_flat,
        )
        semantic_masks = semantic_masks_flat.reshape(b, c, h, w)

        return semantic_masks  # (b, c, h, w)

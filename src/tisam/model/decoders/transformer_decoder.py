"""
Mask2Former transformer decoder built on the local masked attention layers.

This module depends on `model.layers.MaskedMultiScaleAttention` for decoder
cross-attention and is orchestrated by `model.decoders.mask2former_decoder`.
`model.segmentor` passes `use_pos_embed` through this decoder so the image-token
attention path can apply axial RoPE from the feature-map geometry alone.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from prettyterm import get_logger

from ..feature_pyramid import SAM3_FEATURE_PYRAMID_SPEC, SpatialShape
from ..layers import MaskedMultiScaleAttention, build_axial_rope_frequencies
from ..positional_embedding import DecoderPositionProvider, resolve_pos_embed_mode
from ..predictors import MaskPredictor

logger = get_logger(__name__)

# Legacy direct constructors and compile-metadata checks patch this alias.
# Segmentor construction passes encoder-specific shapes explicitly.
_FPN_LEVEL_SIZES = SAM3_FEATURE_PYRAMID_SPEC.spatial_shapes


class Mask2FormerDecoderLayer(nn.Module):
    """
    Single Mask2Former decoder layer with masked cross-attention.

    `Mask2FormerTransformerDecoder` instantiates this layer, and the layer calls
    `MaskedMultiScaleAttention` for query-to-image fusion over one FPN level.

    Each layer:
    1. Applies self-attention on queries
    2. Applies masked cross-attention to multi-scale features
    3. Applies FFN
    4. Predicts masks for the next layer to use

    Args:
        d_model: Transformer dimension
        n_heads: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout probability
        hard_attn: Whether masks are treated as hard gates.
        use_sdpa_attn: Whether masked cross-attention uses PyTorch SDPA.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        hard_attn: bool = False,
        use_sdpa_attn: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Masked cross-attention to single scale
        self.cross_attn = MaskedMultiScaleAttention(
            d_model,
            n_heads,
            dropout,
            hard_attn=hard_attn,
            use_sdpa_attn=use_sdpa_attn,
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def with_pos_embed(self, tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        """Add positional encoding to tensor."""
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        prev_masks: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        cross_attn_query_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
        use_pos_embed: bool = False,
        rope_freqs: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of Mask2Former decoder layer.

        Args:
            query: (n, b, d_model) input query embeddings
            memory: (b, c, h, w) single-scale feature map to attend to
            prev_masks: (b, n, h, w) optional masks from previous layer for gating
            query_pos: (n, b, d_model) optional query positional encoding
            cross_attn_query_pos: optional query positions used only for cross-attention
            memory_pos: (b, d_model, h, w) optional image positional encoding
            use_pos_embed: Whether to apply axial RoPE in masked cross-attention.
            rope_freqs: Precomputed axial frequencies for the current FPN level.
            attn_mask: (n, h*w) optional attention mask

        Returns:
            query_out: (n, b, d_model) refined query embeddings
        """
        # === Self-attention ===
        # query: (n, b, d_model) with seq_first format for nn.MultiheadAttention
        q = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(q, q, query)[0]
        query = query + self.self_attn_dropout(query2)
        query = self.self_attn_norm(query)

        # === Masked cross-attention ===
        # Uses prev_masks to gate attention
        query2 = self.cross_attn(
            query=query,
            memory=memory,
            mask=prev_masks,  # Mask from previous layer gates attention
            attn_mask=attn_mask,
            use_pos_embed=use_pos_embed,
            query_pos=cross_attn_query_pos,
            memory_pos=memory_pos,
            rope_freqs=rope_freqs,
        )
        query = query + self.cross_attn_dropout(query2)
        query = self.cross_attn_norm(query)

        # === FFN ===
        query2 = self.ffn(query)
        query = query + self.ffn_dropout(query2)
        query = self.ffn_norm(query)

        return query


class Mask2FormerTransformerDecoder(nn.Module):
    """
    Mask2Former transformer decoder with masked multi-scale attention.

    Key differences from DETR:
    - Uses PREDICTED MASKS to gate cross-attention (not reference boxes)
    - Each decoder layer attends to ONE feature scale (cycles through scales)
    - Returns true intermediate outputs (not duplicated copies)
    - Predicts masks at each layer for deep supervision

    Args:
        d_model: Transformer dimension
        num_layers: Number of decoder layers
        n_heads: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout probability
        num_queries: Number of learnable object queries
        num_feature_levels: Number of FPN levels (typically 4)
        mask_predictor: Optional external MaskPredictor to use for attention gating
        use_pos_embed: Whether decoder cross-attention should apply axial RoPE to
            image-side keys using feature-map geometry.
        hard_attn: Whether masks are treated as hard gates.
        use_sdpa_attn: Whether masked cross-attention uses PyTorch SDPA.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_queries: int = 33,
        num_feature_levels: int = 4,
        mask_predictor: MaskPredictor | None = None,
        hard_attn: bool = False,
        use_sdpa_attn: bool = True,
        use_pos_embed: bool = True,
        pos_embed_mode: str | None = None,
        learned_pos_embed_init: str = "normal",
        learned_pos_embed_std: float = 0.02,
        learned_image_pos_scope: str = "per_level",
        learned_query_pos: bool = True,
        learned_image_pos: bool = True,
        feature_shapes: tuple[SpatialShape, ...] | None = None,
    ):
        super().__init__()
        feature_shapes = feature_shapes or _FPN_LEVEL_SIZES
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.feature_shapes = feature_shapes
        if len(feature_shapes) != num_feature_levels:
            raise ValueError("Transformer feature-shape count must match num_feature_levels.")
        self.use_pos_embed = use_pos_embed
        self.hard_attn = hard_attn
        self.use_sdpa_attn = use_sdpa_attn
        self.pos_embed_mode = resolve_pos_embed_mode(
            use_pos_embed=use_pos_embed,
            pos_embed_mode=pos_embed_mode,
        )
        self.use_rope = self.pos_embed_mode == "rope"
        self.rope_frequencies_0: torch.Tensor
        self.rope_frequencies_1: torch.Tensor
        self.rope_frequencies_2: torch.Tensor
        self.rope_frequencies_3: torch.Tensor
        head_dim = d_model // n_heads
        for level_idx, (height, width) in enumerate(feature_shapes):
            frequencies = (
                build_axial_rope_frequencies(height, width, head_dim)
                if self.use_rope
                else torch.empty(0, dtype=torch.complex64)
            )
            self.register_buffer(
                f"rope_frequencies_{level_idx}",
                frequencies,
                persistent=False,
            )

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, d_model)
        nn.init.normal_(self.query_embed.weight)
        self.query_pos_embed: nn.Parameter | None = None

        # Custom decoder layers with masked cross-attention
        self.layers = nn.ModuleList(
            [
                Mask2FormerDecoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    hard_attn=hard_attn,
                    use_sdpa_attn=use_sdpa_attn,
                )
                for _ in range(num_layers)
            ]
        )

        # Scale cycling: which layer attends to which scale
        # For 6 layers and 4 scales: [0, 1, 2, 3, 0, 1]
        # This ensures each layer processes features at a different resolution
        self.scale_cycle = [i % num_feature_levels for i in range(num_layers)]

        self.position_provider: DecoderPositionProvider | None = None
        if self.pos_embed_mode == "learned":
            self.position_provider = DecoderPositionProvider(
                d_model=d_model,
                num_queries=num_queries,
                num_feature_levels=num_feature_levels,
                query_enabled=learned_query_pos,
                image_enabled=learned_image_pos,
                init=learned_pos_embed_init,  # type: ignore[arg-type]
                init_std=learned_pos_embed_std,
                image_scope=learned_image_pos_scope,  # type: ignore[arg-type]
                image_shapes=feature_shapes,
            )
        else:
            # Legacy no-PE and RoPE checkpoints expect this trainable key.
            self.query_pos_embed = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)

        # Use provided mask_predictor or create default
        if mask_predictor is None:
            self.mask_predictor = MaskPredictor(d_model=d_model, num_layers=3)
        else:
            self.mask_predictor = mask_predictor

        logger.success(
            f"Mask2FormerTransformerDecoder initialized: "
            f"d_model={d_model}, num_layers={num_layers}, n_heads={n_heads}, "
            f"num_queries={num_queries}, num_feature_levels={num_feature_levels}, "
            f"scale_cycle={self.scale_cycle}, "
            f"use_pos_embed={use_pos_embed}, "
            f"use_sdpa_attn={use_sdpa_attn}, "
            f"pos_embed_mode={self.pos_embed_mode}, "
            f"shared_mask_predictor={mask_predictor is not None}"
        )

    def prepare_compile_metadata(self) -> None:
        """Verify fixed RoPE buffers after device placement and before compilation."""
        device = self.query_embed.weight.device
        frequencies = (
            self.rope_frequencies_0,
            self.rope_frequencies_1,
            self.rope_frequencies_2,
            self.rope_frequencies_3,
        )
        for level_idx, ((height, width), frequency) in enumerate(
            zip(self.feature_shapes, frequencies, strict=True)
        ):
            expected_shape = (height * width, self.d_model // self.n_heads // 2)
            if self.use_rope and frequency.shape != expected_shape:
                raise RuntimeError(
                    f"Decoder RoPE level {level_idx} has shape {tuple(frequency.shape)}, "
                    f"expected {expected_shape}."
                )
            if frequency.device != device:
                raise RuntimeError(
                    "Decoder RoPE metadata must follow the decoder device before compilation."
                )

    def _validate_features(
        self,
        multi_scale_features: list[torch.Tensor],
        multi_scale_pos: list[torch.Tensor] | None = None,
    ) -> None:
        """Validate feature count for `forward` before scale cycling."""
        if len(multi_scale_features) != self.num_feature_levels:
            raise ValueError(
                f"Expected {self.num_feature_levels} multi_scale_features, got "
                f"{len(multi_scale_features)}"
            )
        if multi_scale_pos is None:
            return
        if len(multi_scale_pos) != len(multi_scale_features):
            raise ValueError(
                f"Expected {len(multi_scale_features)} positional feature levels, got "
                f"{len(multi_scale_pos)}."
            )
        for feature, position in zip(multi_scale_features, multi_scale_pos, strict=True):
            if feature.shape != position.shape:
                raise ValueError(
                    "Image positional tensors must match feature shapes; "
                    f"got feature={tuple(feature.shape)}, pos={tuple(position.shape)}."
                )

    def image_positions(
        self,
        multi_scale_features: list[torch.Tensor],
    ) -> list[torch.Tensor] | None:
        """Return learned image positions generated after `PixelDecoder.forward`."""
        if self.position_provider is None:
            return None
        return self.position_provider.image_positions(multi_scale_features)

    def _initial_queries(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Create batched query tensors for `forward`."""
        query_embed = self.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        if self.position_provider is not None:
            query_pos = self.position_provider.query_positions(batch_size, reference=tgt)
        else:
            if self.query_pos_embed is None:
                return tgt, None
            query_pos = self.query_pos_embed.unsqueeze(1).repeat(1, batch_size, 1)
        return tgt, query_pos

    @staticmethod
    def _resize_prev_masks(
        prev_masks: torch.Tensor | None,
        target_hw: tuple[int, int],
    ) -> torch.Tensor | None:
        """Resize previous masks to the current feature scale for `forward`."""
        if prev_masks is None:
            return None
        return torch.nn.functional.interpolate(
            prev_masks,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )

    def _attention_masks(self, prev_mask_logits: torch.Tensor | None) -> torch.Tensor | None:
        """Convert `MaskPredictor` logits into masks consumed by decoder attention."""
        if prev_mask_logits is None or self.hard_attn:
            return prev_mask_logits
        return prev_mask_logits.sigmoid()

    def _decode_layer(
        self,
        layer_idx: int,
        layer: Mask2FormerDecoderLayer,
        tgt: torch.Tensor,
        query_pos: torch.Tensor | None,
        prev_masks: torch.Tensor | None,
        multi_scale_features: list[torch.Tensor],
        multi_scale_pos: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one transformer layer and predict masks for the next `forward` layer."""
        scale_idx = self.scale_cycle[layer_idx]
        memory = multi_scale_features[scale_idx]
        memory_pos = multi_scale_pos[scale_idx] if multi_scale_pos is not None else None
        prev_masks = self._resize_prev_masks(prev_masks, self.feature_shapes[scale_idx])
        attention_masks = self._attention_masks(prev_masks)
        rope_frequencies = (
            (
                self.rope_frequencies_0,
                self.rope_frequencies_1,
                self.rope_frequencies_2,
                self.rope_frequencies_3,
            )[scale_idx]
            if self.use_rope
            else None
        )

        tgt = layer(
            query=tgt,
            memory=memory,
            prev_masks=attention_masks,
            query_pos=query_pos,
            cross_attn_query_pos=query_pos if self.pos_embed_mode == "learned" else None,
            memory_pos=memory_pos,
            use_pos_embed=self.use_rope,
            rope_freqs=rope_frequencies,
        )
        return tgt, self.mask_predictor(tgt, memory)

    def forward(
        self,
        multi_scale_features: list[torch.Tensor],
        multi_scale_pos: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through Mask2Former transformer decoder.

        Args:
            multi_scale_features: List of (b, d_model, h, w) multi-scale features
                ordered from P2 to P5.
            multi_scale_pos: Optional learned image positions with the same shapes.

        Returns:
            hs: (num_layers, n, b, d_model) intermediate query embeddings
        """
        self._validate_features(multi_scale_features, multi_scale_pos)
        b = multi_scale_features[0].shape[0]
        tgt, query_pos = self._initial_queries(b)
        all_hs = tgt.new_empty((self.num_layers, self.num_queries, b, self.d_model))
        prev_masks: torch.Tensor | None = None
        for layer_idx, layer in enumerate(self.layers):
            tgt, prev_masks = self._decode_layer(
                layer_idx,
                cast(Mask2FormerDecoderLayer, layer),
                tgt,
                query_pos,
                prev_masks,
                multi_scale_features,
                multi_scale_pos,
            )
            all_hs[layer_idx] = tgt
        return all_hs

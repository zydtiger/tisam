"""Learnable decoder positional embeddings for TiSAM.

`tisam.checkpointing` and `model.segmentor` thread positional embedding
configuration into `model.decoders.transformer_decoder`. That decoder calls the
providers in this module to create query and image-feature positional tensors
before `model.layers.MaskedMultiScaleAttention` consumes them.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn as nn

from .feature_pyramid import SAM3_FEATURE_PYRAMID_SPEC, SpatialShape

PositionalEmbeddingMode = Literal["none", "rope", "learned"]
LearnedPositionalEmbeddingInit = Literal["normal"]
LearnedImagePositionScope = Literal["per_level"]

POSITIONAL_EMBEDDING_MODES: frozenset[str] = frozenset(("none", "rope", "learned"))
LEARNED_IMAGE_POSITION_SHAPES = SAM3_FEATURE_PYRAMID_SPEC.spatial_shapes


def resolve_pos_embed_mode(
    *,
    use_pos_embed: bool,
    pos_embed_mode: str | None,
) -> PositionalEmbeddingMode:
    """Resolve legacy boolean and mode-based positional config.

    `Mask2FormerSegmentor` and `Mask2FormerTransformerDecoder` call this while
    preserving old direct constructor calls that pass only `use_pos_embed`.
    """
    if pos_embed_mode is None:
        return "rope" if use_pos_embed else "none"
    if pos_embed_mode not in POSITIONAL_EMBEDDING_MODES:
        raise ValueError(
            f"Unknown positional embedding mode '{pos_embed_mode}'. "
            "Expected one of: none, rope, learned."
        )
    if (pos_embed_mode != "none") != use_pos_embed:
        raise ValueError(
            "`use_pos_embed` must be false for positional mode 'none' and true otherwise."
        )
    return cast(PositionalEmbeddingMode, pos_embed_mode)


class LearnableQueryPosition(nn.Module):
    """Learn query-index positions consumed by the transformer decoder.

    `DecoderPositionProvider` owns this module and
    `Mask2FormerTransformerDecoder` expands its output to match the active
    batch before self-attention and cross-attention use the query positions.
    """

    def __init__(
        self,
        num_queries: int,
        d_model: int,
        *,
        init: LearnedPositionalEmbeddingInit = "normal",
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if init != "normal":
            raise ValueError(f"Unsupported query positional initialization: {init}")
        if init_std <= 0:
            raise ValueError("init_std must be positive.")

        self.weight = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.normal_(self.weight, std=init_std)

    def forward(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        """Return positions shaped like decoder query tensors."""
        return (
            self.weight.to(device=reference.device, dtype=reference.dtype)
            .unsqueeze(1)
            .expand(
                -1,
                batch_size,
                -1,
            )
        )


class LearnableImagePosition(nn.Module):
    """Learn per-level image positions for pixel-decoder feature maps.

    `DecoderPositionProvider` calls this after `PixelDecoder.forward`, and the
    resulting tensors are passed through `Mask2FormerTransformerDecoder` to
    `MaskedMultiScaleAttention` for image-side key positioning.
    """

    def __init__(
        self,
        d_model: int,
        num_feature_levels: int,
        *,
        image_shapes: tuple[SpatialShape, ...] | None = None,
        scope: LearnedImagePositionScope = "per_level",
        init: LearnedPositionalEmbeddingInit = "normal",
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if scope != "per_level":
            raise ValueError(f"Unsupported learned image positional scope: {scope}")
        if init != "normal":
            raise ValueError(f"Unsupported image positional initialization: {init}")
        if init_std <= 0:
            raise ValueError("init_std must be positive.")

        self.d_model = d_model
        self.image_shapes = image_shapes or self.base_shapes(num_feature_levels)
        if len(self.image_shapes) != num_feature_levels:
            raise ValueError("Learned image-position shapes must match num_feature_levels.")
        self.level_embeddings = nn.ParameterList()
        for height, width in self.image_shapes:
            weight = nn.Parameter(torch.empty(1, d_model, height, width))
            nn.init.normal_(weight, std=init_std)
            self.level_embeddings.append(weight)

    @staticmethod
    def base_shapes(num_feature_levels: int) -> tuple[tuple[int, int], ...]:
        """Return canonical FPN shapes used to initialize learned image maps."""
        if num_feature_levels <= len(LEARNED_IMAGE_POSITION_SHAPES):
            return LEARNED_IMAGE_POSITION_SHAPES[:num_feature_levels]
        tail = (LEARNED_IMAGE_POSITION_SHAPES[-1],) * (
            num_feature_levels - len(LEARNED_IMAGE_POSITION_SHAPES)
        )
        return LEARNED_IMAGE_POSITION_SHAPES + tail

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Return one image position tensor for each decoder feature level."""
        if len(features) != len(self.level_embeddings):
            raise ValueError(
                f"Expected {len(self.level_embeddings)} feature levels, got {len(features)}."
            )

        positions: list[torch.Tensor] = []
        for level_idx, (feature, weight) in enumerate(
            zip(features, self.level_embeddings, strict=True)
        ):
            if feature.shape[1] != self.d_model:
                raise ValueError(
                    f"Expected feature channels={self.d_model}, got {feature.shape[1]}."
                )
            target_hw = self.image_shapes[level_idx]
            if feature.shape[-2:] != target_hw:
                raise ValueError(
                    f"Expected feature level {level_idx} shape {target_hw}, "
                    f"got {tuple(feature.shape[-2:])}."
                )
            pos = weight.to(device=feature.device, dtype=feature.dtype)
            if pos.shape[-2:] != target_hw:
                pos = torch.nn.functional.interpolate(
                    pos,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            positions.append(pos.expand(feature.shape[0], -1, -1, -1))
        return positions


class DecoderPositionProvider(nn.Module):
    """Create learned query and image positions for the transformer decoder.

    `Mask2FormerTransformerDecoder` owns this provider in learned mode. Its
    query output is consumed by decoder self-attention and cross-attention, and
    its image outputs are consumed by the cross-attention path after
    `Mask2FormerDecoder` obtains multi-scale image features.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_queries: int,
        num_feature_levels: int,
        query_enabled: bool = True,
        image_enabled: bool = True,
        init: LearnedPositionalEmbeddingInit = "normal",
        init_std: float = 0.02,
        image_scope: LearnedImagePositionScope = "per_level",
        image_shapes: tuple[SpatialShape, ...] | None = None,
    ) -> None:
        super().__init__()
        if not query_enabled and not image_enabled:
            raise ValueError("At least one learned positional path must be enabled.")

        self.query_pos = (
            LearnableQueryPosition(
                num_queries=num_queries,
                d_model=d_model,
                init=init,
                init_std=init_std,
            )
            if query_enabled
            else None
        )
        self.image_pos = (
            LearnableImagePosition(
                d_model=d_model,
                num_feature_levels=num_feature_levels,
                image_shapes=image_shapes,
                scope=image_scope,
                init=init,
                init_std=init_std,
            )
            if image_enabled
            else None
        )

    def query_positions(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor | None:
        """Return learned query positions for `Mask2FormerTransformerDecoder`."""
        if self.query_pos is None:
            return None
        return self.query_pos(batch_size=batch_size, reference=reference)

    def image_positions(self, features: list[torch.Tensor]) -> list[torch.Tensor] | None:
        """Return learned image positions for `Mask2FormerDecoder.forward`."""
        if self.image_pos is None:
            return None
        return self.image_pos(features)

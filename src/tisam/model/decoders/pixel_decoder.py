"""
Pixel decoder component for SAM3 Mask2Former decoder.

FPN-style upsampling to combine multi-scale backbone features into
high-resolution pixel embeddings for mask prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from prettyterm import get_logger

from ..encoders.extra_encoder import ExtraFeaturePyramid, ExtraFeatures
from ..feature_pyramid import SAM3_FEATURE_PYRAMID_SPEC, FeaturePyramidSpec
from .deform_attn import DeformableCrossAttention

logger = get_logger(__name__)

# Preserve direct-constructor compatibility while production model construction
# passes an explicit immutable `FeaturePyramidSpec`.
_FPN_LEVEL_SIZES = SAM3_FEATURE_PYRAMID_SPEC.spatial_shapes
_FPN_TOP_DOWN_TARGET_SIZES = SAM3_FEATURE_PYRAMID_SPEC.top_down_target_shapes


class PixelDecoder(nn.Module):
    """
    FPN-style pixel decoder for upsampling multi-scale features.

    Combines backbone FPN features from multiple scales into a single
    high-resolution feature map for mask prediction.

    Architecture:
        backbone_fpn: [(b,c2,h2,w2), (b,c3,h3,w3), (b,c4,h4,w4), (b,c5,h5,w5)]
            where [c2, c3, c4, c5] = fpn_channels (not required to equal d_model)
            ↓
        extra_features: optional final map or shallow/middle/final map tuple from ExtraEncoder
            ↓
        fuse P3/P4/P5 backbone features (channels from fpn_channels) with
        their mapped extra features to eventually produce d_model-channel mask features
            ↓
        top-down fusion with upsampling (start from lowest res, upsample and add)
            ↓
        conv + norm + relu at each stage
            ↓
        pixel_embed: (b, d_model, output_h, output_w)

    Based on SAM3's PixelDecoder in libs/sam3/sam3/model/maskformer_segmentation.py
    """

    def __init__(
        self,
        d_model: int = 256,
        fpn_channels: list[int] | None = None,
        upsampling_stages: int = 3,
        interpolation_mode: str = "nearest",
        shared_conv: bool = False,
        extra_embed_dim: int | None = 1280,
        use_cross_attn: bool = False,
        output_hw: tuple[int, int] | None = None,
        sam_proj: bool = False,
        early_interpolation: bool = True,
        feature_pyramid_spec: FeaturePyramidSpec | None = None,
    ):
        super().__init__()
        feature_pyramid_spec = feature_pyramid_spec or FeaturePyramidSpec(
            channels=SAM3_FEATURE_PYRAMID_SPEC.channels,
            spatial_shapes=_FPN_LEVEL_SIZES,
        )
        self.d_model = d_model
        self.feature_pyramid_spec = feature_pyramid_spec
        self.fpn_channels = fpn_channels or [d_model] * len(feature_pyramid_spec.channels)
        self.upsampling_stages = upsampling_stages
        self.interpolation_mode = interpolation_mode
        self.shared_conv = shared_conv
        self.extra_embed_dim = extra_embed_dim
        self.use_extra_fusion = extra_embed_dim is not None
        self.use_cross_attn = use_cross_attn
        self.output_hw = output_hw
        self.sam_proj = sam_proj
        self.early_interpolation = early_interpolation

        self.fpn_input_proj = self._build_fpn_projection(d_model)
        self.conv_layers, self.norms = self._build_upsampling_layers(
            d_model,
            shared_conv,
            upsampling_stages,
        )
        self.output_refine_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        spatial_shapes = torch.tensor(feature_pyramid_spec.spatial_shapes, dtype=torch.long)
        level_sizes = spatial_shapes.prod(dim=1)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), level_sizes.cumsum(dim=0)[:-1])
        )
        self.spatial_shapes: torch.Tensor
        self.level_start_index: torch.Tensor
        self.register_buffer("spatial_shapes", spatial_shapes, persistent=False)
        self.register_buffer("level_start_index", level_start_index, persistent=False)
        self._init_extra_fusion_layers(d_model)
        self._log_initialization(d_model, upsampling_stages, shared_conv, extra_embed_dim)

    def _build_fpn_projection(self, d_model: int) -> nn.ModuleList | None:
        """Build optional FPN channel projections consumed by `forward`."""
        if not self.sam_proj and all(channel == d_model for channel in self.fpn_channels):
            return None
        return nn.ModuleList(
            [nn.Conv2d(in_channels, d_model, kernel_size=1) for in_channels in self.fpn_channels]
        )

    @staticmethod
    def _build_upsampling_layers(
        d_model: int,
        shared_conv: bool,
        upsampling_stages: int,
    ) -> tuple[nn.ModuleList, nn.ModuleList]:
        """Build top-down conv/norm stages consumed by `forward`."""
        num_convs = 1 if shared_conv else upsampling_stages
        conv_layers = []
        norms = []
        for _ in range(num_convs):
            conv_layers.append(nn.Conv2d(d_model, d_model, 3, padding=1))
            norms.append(nn.GroupNorm(8, d_model))
        return nn.ModuleList(conv_layers), nn.ModuleList(norms)

    def _init_extra_fusion_layers(self, d_model: int) -> None:
        """Initialize optional extra-encoder fusion layers consumed by `forward`."""
        if self.use_extra_fusion:
            extra_dim = self.extra_embed_dim
            assert extra_dim is not None
            if self.use_cross_attn:
                self.deform_attn5 = DeformableCrossAttention(
                    embed_dim=d_model,
                    num_heads=8,
                    num_points=4,
                    value_dim=extra_dim,
                )
                self.deform_attn4 = DeformableCrossAttention(
                    embed_dim=d_model,
                    num_heads=8,
                    num_points=4,
                    value_dim=extra_dim,
                )
                self.deform_attn3 = DeformableCrossAttention(
                    embed_dim=d_model,
                    num_heads=8,
                    num_points=4,
                    value_dim=extra_dim,
                )
            else:
                # Projection layers for extra encoder features (e.g., Virchow2: 1280 -> 256)
                self.extra_proj5 = nn.Conv2d(extra_dim, d_model, 1)
                self.extra_proj4 = nn.Conv2d(extra_dim, d_model, 1)
                self.extra_proj3 = nn.Conv2d(extra_dim, d_model, 1)

                # Spatial attentions for extra feature fusion
                self.attn_conv5 = nn.Conv2d(d_model * 2, 1, kernel_size=1)
                self.attn_conv4 = nn.Conv2d(d_model * 2, 1, kernel_size=1)
                self.attn_conv3 = nn.Conv2d(d_model * 2, 1, kernel_size=1)

    def _log_initialization(
        self,
        d_model: int,
        upsampling_stages: int,
        shared_conv: bool,
        extra_embed_dim: int | None,
    ) -> None:
        """Log pixel-decoder settings for model construction diagnostics."""
        logger.success(
            f"PixelDecoder initialized: "
            f"d_model={d_model}, upsampling_stages={upsampling_stages}, "
            f"fpn_channels={self.fpn_channels}, "
            f"shared_conv={shared_conv}, sam_proj={self.sam_proj}, "
            f"use_extra_fusion={extra_embed_dim is not None}, "
            f"use_cross_attn={self.use_cross_attn}, extra_embed_dim={extra_embed_dim}, "
            f"output_hw={self.output_hw}"
        )

    def _prepare_backbone_fpn(self, backbone_fpn: list[torch.Tensor]) -> list[torch.Tensor]:
        """Validate and normalize backbone features for `forward`.

        `forward` calls this before extra-encoder fusion so downstream
        Mask2Former decoder components always receive `d_model` channels.
        """
        normalized_fpn = list(backbone_fpn)
        if len(normalized_fpn) != len(self.fpn_channels):
            raise ValueError(
                f"Expected {len(self.fpn_channels)} backbone_fpn levels, got {len(normalized_fpn)}"
            )

        if self.fpn_input_proj is None:
            return normalized_fpn
        return [proj(feat) for proj, feat in zip(self.fpn_input_proj, normalized_fpn)]

    def _fuse_cross_attention_extra_features(
        self,
        backbone_fpn: list[torch.Tensor],
        extra_features: ExtraFeaturePyramid,
    ) -> None:
        """Fuse extra encoder features through deformable attention for `forward`.

        `forward` applies this path when `Mask2FormerSegmentor` enables
        cross-attention fusion between SAM3 FPN levels and extra encoder output.
        """
        extra_p3, extra_p4, extra_p5 = extra_features

        p5 = backbone_fpn[3]
        backbone_fpn[3] = p5 + self.deform_attn5(query=p5, value=extra_p5)

        p4 = backbone_fpn[2]
        backbone_fpn[2] = p4 + self.deform_attn4(query=p4, value=extra_p4)

        p3 = backbone_fpn[1]
        backbone_fpn[1] = p3 + self.deform_attn3(query=p3, value=extra_p3)

    def _fuse_spatial_extra_features(
        self,
        backbone_fpn: list[torch.Tensor],
        extra_features: ExtraFeaturePyramid,
    ) -> None:
        """Fuse extra encoder features through spatial attention for `forward`.

        `forward` applies this default path before top-down FPN fusion, and the
        mask decoder consumes the resulting multi-scale features.
        """
        extra_p3_source, extra_p4_source, extra_p5_source = extra_features
        extra_proj5 = self.extra_proj5(extra_p5_source)
        extra_proj4 = self.extra_proj4(extra_p4_source)
        extra_proj3 = self.extra_proj3(extra_p3_source)

        p5 = backbone_fpn[3]
        extra_p5 = torch.nn.functional.interpolate(
            extra_proj5,
            size=self.feature_pyramid_spec.spatial_shapes[3],
            mode="bilinear",
            align_corners=False,
        )
        concat_p5 = torch.cat([p5, extra_p5], dim=1)
        attn_p5 = torch.sigmoid(self.attn_conv5(concat_p5))
        backbone_fpn[3] = attn_p5 * p5 + (1 - attn_p5) * extra_p5

        p4 = backbone_fpn[2]
        extra_p4 = torch.nn.functional.interpolate(
            extra_proj4,
            size=self.feature_pyramid_spec.spatial_shapes[2],
            mode="bilinear",
            align_corners=False,
        )
        concat_p4 = torch.cat([p4, extra_p4], dim=1)
        attn_p4 = torch.sigmoid(self.attn_conv4(concat_p4))
        backbone_fpn[2] = attn_p4 * p4 + (1 - attn_p4) * extra_p4

        p3 = backbone_fpn[1]
        extra_p3 = torch.nn.functional.interpolate(
            extra_proj3,
            size=self.feature_pyramid_spec.spatial_shapes[1],
            mode="bilinear",
            align_corners=False,
        )
        concat_p3 = torch.cat([p3, extra_p3], dim=1)
        attn_p3 = torch.sigmoid(self.attn_conv3(concat_p3))
        backbone_fpn[1] = attn_p3 * p3 + (1 - attn_p3) * extra_p3

    @staticmethod
    def _resolve_extra_feature_pyramid(extra_features: ExtraFeatures) -> ExtraFeaturePyramid:
        """Map legacy final features or deep features onto P3/P4/P5 fusion inputs."""
        if isinstance(extra_features, torch.Tensor):
            return extra_features, extra_features, extra_features
        if len(extra_features) != 3:
            raise ValueError(f"Expected three deep extra features, got {len(extra_features)}")
        return extra_features

    def _fuse_extra_features(
        self,
        backbone_fpn: list[torch.Tensor],
        extra_features: ExtraFeatures | None,
    ) -> None:
        """Apply optional extra encoder fusion for `forward`.

        `Mask2FormerSegmentor.forward` passes extra encoder output here when
        configured; no-op behavior preserves SAM3-only decoding.
        """
        if extra_features is None:
            return
        if not self.use_extra_fusion:
            raise ValueError("Received extra_features, but PixelDecoder was built without fusion.")

        extra_feature_pyramid = self._resolve_extra_feature_pyramid(extra_features)
        if self.use_cross_attn:
            self._fuse_cross_attention_extra_features(backbone_fpn, extra_feature_pyramid)
        else:
            self._fuse_spatial_extra_features(backbone_fpn, extra_feature_pyramid)

    def _top_down_fuse(self, backbone_fpn: list[torch.Tensor]) -> torch.Tensor:
        """Run FPN top-down upsampling for `forward`.

        The returned pixel embedding is refined into mask features consumed by
        `Mask2FormerDecoder`.
        """
        prev_fpn = backbone_fpn[-1]
        fpn_feats = backbone_fpn[:-1]

        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            upsampled = torch.nn.functional.interpolate(
                prev_fpn,
                size=self.feature_pyramid_spec.top_down_target_shapes[layer_idx],
                mode=self.interpolation_mode,
            )
            prev_fpn = bb_feat + upsampled

            conv_idx = 0 if self.shared_conv else layer_idx
            prev_fpn = self.conv_layers[conv_idx](prev_fpn)
            prev_fpn = torch.nn.functional.gelu(self.norms[conv_idx](prev_fpn))

        return prev_fpn

    def _refine_pixel_embedding(self, pixel_embed: torch.Tensor) -> torch.Tensor:
        """Resize and locally refine mask features for `forward`.

        `forward` returns this tensor as `mask_features` for downstream mask
        prediction in `Mask2FormerDecoder`.
        """
        if self.early_interpolation and self.output_hw is not None:
            pixel_embed = torch.nn.functional.interpolate(
                pixel_embed,
                size=self.output_hw,
                mode="bilinear",
                align_corners=False,
            )

        pixel_embed = self.output_refine_conv(pixel_embed)
        pixel_embed = torch.nn.functional.gelu(pixel_embed)
        return pixel_embed

    def forward(
        self,
        backbone_fpn: list[torch.Tensor],
        extra_features: ExtraFeatures | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Upsample and fuse multi-scale FPN features.

        Args:
            backbone_fpn: List of FPN features at multiple scales
                [(b, c2, h2, w2), (b, c3, h3, w3), (b, c4, h4, w4), (b, c5, h5, w5)],
                ordered from highest resolution (P2) to lowest (P5), where
                [c2, c3, c4, c5] should follow self.fpn_channels and can be different from d_model.
            extra_features: Optional final histopathology map or a P3/P4/P5-ordered
                tuple of deep maps from `ExtraEncoder`, each shaped
                `(b, extra_embed_dim, 64, 64)`.

        Returns:
            dict with:
                - mask_features: (b, d_model, output_h, output_w) high-res pixel embeddings
                  for mask prediction
                - multi_scale_features: list of (b, c, h, w) multi-scale features for transformer
                - spatial_shapes: (num_levels, 2) tensor of [h, w] per level
                - level_start_index: (num_levels,) tensor for flattening multi-scale features
        """
        backbone_fpn = self._prepare_backbone_fpn(backbone_fpn)
        self._fuse_extra_features(backbone_fpn, extra_features)
        pixel_embed = self._refine_pixel_embedding(self._top_down_fuse(backbone_fpn))

        return {
            "mask_features": pixel_embed,
            "multi_scale_features": backbone_fpn,
            "spatial_shapes": self.spatial_shapes,  # (num_levels, 2)
            "level_start_index": self.level_start_index,  # (num_levels,)
        }

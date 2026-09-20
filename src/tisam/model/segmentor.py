"""Dual-encoder TiSAM model used by the public API, training and inference.

SAM3 spatial features and pathology foundation features feed the TiSAM pixel
and query decoders. Forward returns class logits including the canonical void.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from prettyterm import get_logger

from tisam.config.model import ModelConfig

from .decoders import Mask2FormerDecoder
from .encoders import ExtraEncoder, build_sam3_encoder
from .encoders.extra_encoder import ExtraFeatureMode, ExtraFeatures
from .feature_pyramid import SAM3_FEATURE_PYRAMID_SPEC
from .positional_embedding import resolve_pos_embed_mode

logger = get_logger(__name__)
_CROSS_ATTN_FUSION_PREFIX = "mask_decoder.pixel_decoder.deform_attn"
_POINTWISE_FUSION_PREFIXES = (
    "mask_decoder.pixel_decoder.extra_proj",
    "mask_decoder.pixel_decoder.attn_conv",
)


class TiSAM(nn.Module):
    """Construct dual-encoder TiSAM from a validated architecture configuration."""

    def __init__(self, config: ModelConfig, *, sam3_checkpoint: Path | None = None):
        super().__init__()
        self.config = config
        num_classes = config.num_classes
        assert config.total_queries is not None
        assert config.extra_embed_dim is not None
        total_queries = config.total_queries
        d_model = config.d_model
        num_layers = config.num_layers
        n_heads = config.n_heads
        dim_feedforward = config.dim_feedforward
        dropout = config.dropout
        upsampling_stages = config.upsampling_stages
        finetune = config.finetune
        finetune_last_n_blocks = config.finetune_last_n_blocks
        finetune_neck_convs = config.finetune_neck_convs
        output_hw = config.output_hw
        extra_encoder = config.extra_encoder
        extra_shape = config.extra_shape
        extra_embed_dim = config.extra_embed_dim
        extra_finetune = config.extra_finetune
        extra_finetune_last_n_blocks = config.extra_finetune_last_n_blocks
        extra_feature_mode = config.extra_feature_mode
        pos_embed_mode = config.pos_embed_mode
        learned_pos_embed_init = config.learned_pos_embed_init
        learned_pos_embed_std = config.learned_pos_embed_std
        learned_image_pos_scope = config.learned_image_pos_scope
        learned_query_pos = config.learned_query_pos
        learned_image_pos = config.learned_image_pos
        use_cross_attn = config.use_cross_attn
        hard_attn = config.hard_attn
        use_sdpa_attn = config.use_sdpa_attn
        sam_proj = config.sam_proj
        early_interpolation = config.early_interpolation
        use_pos_embed = config.pos_embed_mode != "none"
        self.num_classes = num_classes
        self.total_classes = num_classes + 1  # +1 for void/background
        self.total_queries = total_queries
        self.output_hw = output_hw
        self.input_hw = output_hw
        self.image_encoder_type = "sam3"
        self.use_sam3_encoder = self.image_encoder_type == "sam3"
        self.feature_pyramid_spec = SAM3_FEATURE_PYRAMID_SPEC
        self.extra_encoder_name = extra_encoder
        self.extra_feature_mode = extra_feature_mode
        self.use_pos_embed = use_pos_embed
        self.pos_embed_mode = resolve_pos_embed_mode(
            use_pos_embed=use_pos_embed,
            pos_embed_mode=pos_embed_mode,
        )
        self.use_cross_attn = use_cross_attn
        self.hard_attn = hard_attn
        self.use_sdpa_attn = use_sdpa_attn
        self.early_interpolation = early_interpolation

        self.image_encoder = build_sam3_encoder(
            sam3_checkpoint=sam3_checkpoint,
            device="cpu",
            finetune=finetune,
            finetune_last_n_blocks=finetune_last_n_blocks,
            finetune_neck_convs=finetune_neck_convs,
        )
        self.image_encoder.eval()
        self.extra_encoder = self._build_extra_encoder(
            extra_encoder,
            extra_shape,
            extra_embed_dim,
            extra_finetune,
            extra_finetune_last_n_blocks,
            extra_feature_mode,
        )
        self.mask_decoder = self._build_mask_decoder(
            num_classes=num_classes,
            total_queries=total_queries,
            d_model=d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            upsampling_stages=upsampling_stages,
            extra_embed_dim=extra_embed_dim,
            use_cross_attn=use_cross_attn,
            hard_attn=hard_attn,
            use_sdpa_attn=use_sdpa_attn,
            sam_proj=sam_proj,
            early_interpolation=early_interpolation,
            use_pos_embed=use_pos_embed,
            pos_embed_mode=self.pos_embed_mode,
            learned_pos_embed_init=learned_pos_embed_init,
            learned_pos_embed_std=learned_pos_embed_std,
            learned_image_pos_scope=learned_image_pos_scope,
            learned_query_pos=learned_query_pos,
            learned_image_pos=learned_image_pos,
        )

    def _build_extra_encoder(
        self,
        extra_encoder: str | None,
        extra_shape: tuple[int, int],
        extra_embed_dim: int,
        extra_finetune: bool,
        extra_finetune_last_n_blocks: int,
        extra_feature_mode: ExtraFeatureMode,
    ) -> ExtraEncoder | None:
        """Build the optional extra encoder consumed by `forward`."""
        if extra_encoder is None:
            return None

        encoder = ExtraEncoder(
            extra_encoder=extra_encoder,
            extra_shape=extra_shape,
            extra_embed_dim=extra_embed_dim,
            finetune=extra_finetune,
            finetune_last_n_blocks=extra_finetune_last_n_blocks,
            feature_mode=extra_feature_mode,
        )
        encoder.configure_freezing()
        encoder.eval()
        return encoder

    def _build_mask_decoder(
        self,
        *,
        num_classes: int,
        total_queries: int,
        d_model: int,
        num_layers: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        upsampling_stages: int,
        extra_embed_dim: int,
        use_cross_attn: bool,
        hard_attn: bool,
        use_sdpa_attn: bool,
        sam_proj: bool,
        early_interpolation: bool,
        use_pos_embed: bool,
        pos_embed_mode: str,
        learned_pos_embed_init: str,
        learned_pos_embed_std: float,
        learned_image_pos_scope: str,
        learned_query_pos: bool,
        learned_image_pos: bool,
    ) -> Mask2FormerDecoder:
        """Build the decoder consumed by `forward`."""
        decoder_extra_embed_dim = (
            extra_embed_dim if self.use_sam3_encoder and self.extra_encoder is not None else None
        )
        return Mask2FormerDecoder(
            num_classes=num_classes,
            total_queries=total_queries,
            d_model=d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            upsampling_stages=upsampling_stages,
            extra_embed_dim=decoder_extra_embed_dim,
            use_cross_attn=use_cross_attn,
            hard_attn=hard_attn,
            use_sdpa_attn=use_sdpa_attn,
            output_hw=self.output_hw,
            sam_proj=sam_proj,
            early_interpolation=early_interpolation,
            use_pos_embed=use_pos_embed,
            pos_embed_mode=pos_embed_mode,
            learned_pos_embed_init=learned_pos_embed_init,
            learned_pos_embed_std=learned_pos_embed_std,
            learned_image_pos_scope=learned_image_pos_scope,
            learned_query_pos=learned_query_pos,
            learned_image_pos=learned_image_pos,
            feature_pyramid_spec=self.feature_pyramid_spec,
        )

    def setup_compile(self, mode: str = "default") -> None:
        """Compile the complete segmentor after validating child static metadata.

        Training, evaluation, and server construction call this method after device
        placement. Compiling the root keeps primary-encoder FPN outputs inside the
        same compiler region as the extra encoder and Mask2Former decoder.
        """
        compile_kwargs = {
            "mode": mode,
            "fullgraph": False,
        }

        if self.image_encoder is not None:
            self.image_encoder.prepare_compile_metadata()
        self.mask_decoder.prepare_compile_metadata()
        self.compile(**compile_kwargs)

    def train(self, mode: bool = True) -> TiSAM:
        """Set train/eval mode while keeping frozen encoders in eval mode."""
        super().train(mode)
        if mode:
            if self.image_encoder is not None and not bool(
                getattr(self.image_encoder, "finetune", False)
            ):
                self.image_encoder.eval()
            if self.extra_encoder is not None and not bool(
                getattr(self.extra_encoder, "finetune", False)
            ):
                self.extra_encoder.eval()
        return self

    def get_optimizer_param_groups(
        self,
        lr: float,
        image_lr_ratio: float = 0.1,
        extra_lr_ratio: float = 0.1,
    ) -> list[dict]:
        """
        Get parameter groups for optimizer with different learning rates.

        Args:
            lr: Base learning rate
            image_lr_ratio: Learning rate ratio for image encoder fine-tuned parameters
            extra_lr_ratio: Learning rate ratio for extra encoder fine-tuned parameters

        Returns:
            List of parameter group dictionaries for optimizer
        """
        # Collect all trainable parameters into separate groups
        base_params = []
        image_finetune_params = []
        extra_finetune_params = []

        # Get fine-tuned parameter names from image encoder
        image_finetuned_names = set()
        if self.image_encoder is not None and self.image_encoder.finetune:
            image_finetuned_names = self.image_encoder.get_finetuned_param_names()

        extra_finetuned_names = set()
        if self.extra_encoder is not None and self.extra_encoder.finetune:
            extra_finetuned_names = self.extra_encoder.get_finetuned_param_names()

        # Categorize parameters
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Check if this is a fine-tuned parameter
            is_image_finetune = any(fn in name for fn in image_finetuned_names)
            is_extra_finetune = any(fn in name for fn in extra_finetuned_names)

            if is_image_finetune:
                image_finetune_params.append(param)
            elif is_extra_finetune:
                extra_finetune_params.append(param)
            else:
                base_params.append(param)

        # Build parameter groups
        param_groups = []

        if base_params:
            param_groups.append({"params": base_params, "lr": lr})
            logger.info(f"Base params: {len(base_params)} parameters with lr={lr}")

        if image_finetune_params:
            param_groups.append({"params": image_finetune_params, "lr": lr * image_lr_ratio})
            logger.info(
                f"Image finetune params: {len(image_finetune_params)} parameters "
                f"with lr={lr * image_lr_ratio}"
            )

        if extra_finetune_params:
            param_groups.append({"params": extra_finetune_params, "lr": lr * extra_lr_ratio})
            logger.info(
                f"Extra finetune params: {len(extra_finetune_params)} parameters "
                f"with lr={lr * extra_lr_ratio}"
            )

        logger.info(
            "Total trainable: "
            f"{len(base_params) + len(image_finetune_params) + len(extra_finetune_params)} "
            f"parameters ({len(base_params)} base, "
            f"{len(image_finetune_params)} image_finetune, "
            f"{len(extra_finetune_params)} extra_finetune)"
        )

        return param_groups

    def _encoder_grad_context(self, encoder: nn.Module | None):
        if encoder is None:
            return nullcontext()
        if bool(getattr(encoder, "finetune", False)) and self.training:
            return torch.enable_grad()
        return torch.no_grad()

    def _validate_checkpoint_fusion_path(self, state_dict) -> None:
        checkpoint_has_cross_attn = any(
            key.startswith(_CROSS_ATTN_FUSION_PREFIX) for key in state_dict
        )
        checkpoint_has_pointwise = any(
            key.startswith(prefix) for prefix in _POINTWISE_FUSION_PREFIXES for key in state_dict
        )

        if self.use_cross_attn:
            if checkpoint_has_pointwise and not checkpoint_has_cross_attn:
                raise RuntimeError(
                    "Checkpoint uses the pointwise extra-feature fusion path, but this model was "
                    "built with `model_use_cross_attn=True`. Load it with "
                    "`model_use_cross_attn=False`."
                )
        elif checkpoint_has_cross_attn:
            raise RuntimeError(
                "Checkpoint uses the deformable cross-attention fusion path, but this model was "
                "built with `model_use_cross_attn=False`. Load it with "
                "`model_use_cross_attn=True`."
            )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self._validate_checkpoint_fusion_path(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self, images: torch.Tensor, return_intermediates: bool = False, return_probs: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass for semantic segmentation.

        Args:
            images: (B, C, H, W) input images
            return_intermediates: If True, return dict with intermediates
            return_probs: If True, return probabilities after softmax (log_softmax + exp)
                          Default is False (returns raw logits)

        Returns:
            - If return_intermediates=False: (B, total_classes, H, W) semantic masks
                - Default: raw logits (unnormalized scores)
                - If return_probs=True: probabilities (after softmax)
            - If return_intermediates=True: dict with:
                - "semantic_masks": (B, total_classes, H, W) final masks (upsampled to output_hw)
                - "decoder_outputs": dict of intermediate outputs at native decoder resolution

        Note:
            - argmax() works identically on logits and probabilities (same ordering)
            - Use return_probs=True for visualization requiring probability values
            - Default logits are suitable for loss computation with cross_entropy
        """

        backbone_fpn = None
        if self.image_encoder is not None:
            with self._encoder_grad_context(self.image_encoder):
                _, backbone_fpn, _, _ = self.image_encoder(images)

        extra_features: ExtraFeatures | None = None
        if self.extra_encoder is not None:
            with self._encoder_grad_context(self.extra_encoder):
                extra_features = self.extra_encoder(images)

        if backbone_fpn is None or extra_features is None:
            raise RuntimeError("Both TiSAM encoders must produce features.")
        decoder_backbone_fpn = backbone_fpn
        decoder_extra_features = extra_features

        # Forward through Mask2Former decoder
        decoder_out = self.mask_decoder(
            backbone_fpn=decoder_backbone_fpn,
            extra_features=decoder_extra_features,
        )

        # Get semantic masks LOGITS at output_hw (B, total_classes, output_H, output_W)
        semantic_masks = decoder_out["semantic_masks"]

        if not self.early_interpolation and semantic_masks.shape[-2:] != self.output_hw:
            semantic_masks = torch.nn.functional.interpolate(
                semantic_masks,
                size=self.output_hw,
                mode="nearest-exact",
            )

        if return_probs:
            semantic_masks = torch.exp(torch.nn.functional.log_softmax(semantic_masks, dim=1))

        if return_intermediates:
            return {
                "semantic_masks": semantic_masks,  # (B, total_classes, output_hw)
                "decoder_outputs": decoder_out,  # mask tensors are already at output_hw
            }

        return semantic_masks

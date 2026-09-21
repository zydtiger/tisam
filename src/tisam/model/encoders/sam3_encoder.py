"""
SAM3 image encoder component.
Wraps SAM3VLBackbone to extract hierarchical vision features.

This encoder provides:
- Low-res features: Deep, semantic-rich features (B, 256, 36, 36)
- High-res features: Detailed, spatial-rich features (B, 256, 288, 288)
- Backbone FPN: Multi-scale feature pyramid
  [(B, 256, 288, 288), (B, 256, 144, 144), (B, 256, 72, 72), (B, 256, 36, 36)]

SAM3 Architecture Notes:
- Input resolution: 1008x1008 (native SAM3 resolution)
- Vision backbone: ViT-H with embed_dim=1024, output_dim=256
- FPN neck: Sam3DualViTDetNeck with scale_factors=[4.0, 2.0, 1.0, 0.5]
- Output features: 4 levels at different resolutions
- Patch grid: 1008/14 = 72x72 patches
- FPN scales applied to patch grid: [72*4=288, 72*2=144, 72*1=72, 72/2=36]
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from prettyterm import get_logger

from tisam.config.weights import (
    SAM3_HF_FILENAME,
    SAM3_HF_REPO_ID,
    SAM3_HF_REVISION,
    SAM3_HF_SHA256,
)

try:
    from sam3.model_builder import build_sam3_image_model
except ImportError:
    build_sam3_image_model = None

logger = get_logger(__name__)

_FPN_LEVEL_SIZES = ((288, 288), (144, 144), (72, 72), (36, 36))


def _pinned_sam3_checkpoint() -> Path:
    """Download and verify the immutable SAM3 weights used by fresh models."""
    checkpoint = Path(
        hf_hub_download(
            repo_id=SAM3_HF_REPO_ID,
            filename=SAM3_HF_FILENAME,
            revision=SAM3_HF_REVISION,
        )
    )
    return verify_sam3_checkpoint(checkpoint)


def verify_sam3_checkpoint(checkpoint: Path) -> Path:
    """Require the same pretrained identity for cached and explicitly supplied weights."""
    digest = hashlib.sha256()
    with checkpoint.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    observed_sha256 = digest.hexdigest()
    if observed_sha256 != SAM3_HF_SHA256:
        raise RuntimeError(
            "Pinned SAM3 checkpoint SHA-256 mismatch: "
            f"expected {SAM3_HF_SHA256}, observed {observed_sha256}."
        )
    return checkpoint


class SAM3Encoder(nn.Module):
    """
    SAM3 perception encoder with vision-language backbone.

    Replaces SAM2 ImageEncoder with SAM3's unified VL architecture.
    Wraps SAM3VLBackbone to extract hierarchical vision features.

    Key differences from SAM2 ImageEncoder:
    - Mid-fusion: Text features can be fused during encoding (not used here)
    - Multi-scale FPN: 4 levels vs SAM2's 3 levels
    - Higher resolution: Native 1008x1008 vs SAM2's 1024x1024
    - Position encodings: Included in backbone output

    Architecture:
    Input (B, 3, 1008, 1008)
        ↓
    ViT-H Backbone (trunk) → 72x72 patch grid
        ↓
    Sam3DualViTDetNeck (FPN with scales [4.0, 2.0, 1.0, 0.5])
        ↓
    Output: backbone_fpn
      [(B, 256, 288, 288), (B, 256, 144, 144), (B, 256, 72, 72), (B, 256, 36, 36)]
    """

    def __init__(
        self,
        sam3_backbone: nn.Module,  # SAM3VLBackbone
        finetune: bool = False,
        finetune_last_n_blocks: int = 2,
        finetune_neck_convs: bool = True,
    ):
        super().__init__()

        # Store configuration
        self.finetune = finetune
        self.finetune_last_n_blocks = finetune_last_n_blocks
        self.finetune_neck_convs = finetune_neck_convs

        # SAM3VLBackbone contains:
        # - vision_backbone: Sam3DualViTDetNeck (ViT + FPN)
        # - language_backbone: VETextEncoder (not used in image-only encoding)
        self.vision_backbone = sam3_backbone.vision_backbone  # Sam3DualViTDetNeck

        # The trunk is the ViT backbone within the neck
        self.trunk = self.vision_backbone.trunk  # type: ignore
        self.position_encoding_0: torch.Tensor
        self.position_encoding_1: torch.Tensor
        self.position_encoding_2: torch.Tensor
        self.position_encoding_3: torch.Tensor
        self._capture_static_position_encodings()

        # Track fine-tuned parameter names for optimizer param groups
        self._finetuned_param_names: set[str] = set()

        logger.success(
            f"SAM3Encoder initialized with "
            f"finetune={finetune}, finetune_last_n_blocks={finetune_last_n_blocks}, "
            f"finetune_neck_convs={finetune_neck_convs}"
        )

    def _capture_static_position_encodings(self) -> None:
        """Move SAM3's fixed positional cache into non-persistent module buffers."""
        if len(self.vision_backbone.convs) != len(_FPN_LEVEL_SIZES):  # type: ignore
            raise ValueError("TiSAM requires the canonical four-level SAM3 FPN.")
        position_encoding = cast(nn.Module, self.vision_backbone.position_encoding)  # type: ignore
        cache = getattr(position_encoding, "cache", None)
        if not isinstance(cache, dict):
            raise TypeError("SAM3 position encoding must expose its precomputed cache.")

        parameter = next(self.vision_backbone.parameters())  # type: ignore
        for level_idx, (height, width) in enumerate(_FPN_LEVEL_SIZES):
            cached = cache.get((height, width))
            if cached is None:
                reference = torch.zeros(
                    (1, 1, height, width),
                    device=parameter.device,
                )
                cached = position_encoding(reference)[0]
            self.register_buffer(
                f"position_encoding_{level_idx}",
                cached.detach().to(device=parameter.device).clone(),
                persistent=False,
            )
        cache.clear()

    def prepare_compile_metadata(self) -> None:
        """Verify fixed positional buffers after device placement and before compilation."""
        device = next(self.vision_backbone.parameters()).device  # type: ignore
        for level_idx in range(len(_FPN_LEVEL_SIZES)):
            position = getattr(self, f"position_encoding_{level_idx}")
            if position.device != device:
                raise RuntimeError(
                    "SAM3 positional metadata must follow the encoder device before compilation."
                )
        cache = getattr(self.vision_backbone.position_encoding, "cache", None)  # type: ignore
        if isinstance(cache, dict):
            cache.clear()

    def configure_freezing(self) -> None:
        """Configure parameter freezing based on fine-tuning settings."""
        if not self.finetune:
            # Freeze all vision backbone parameters
            for param in self.vision_backbone.parameters():  # type: ignore
                param.requires_grad = False
            logger.info("Froze all SAM3 vision backbone parameters")
        else:
            # Get total number of blocks in ViT trunk (32 in SAM3 ViT-H)
            num_blocks = len(self.trunk.blocks)  # type: ignore

            # First freeze all parameters
            for param in self.vision_backbone.parameters():  # type: ignore
                param.requires_grad = False

            # Unfreeze the last N blocks of ViT
            start_block = max(0, num_blocks - self.finetune_last_n_blocks)
            for i in range(start_block, num_blocks):
                block = self.trunk.blocks[i]  # type: ignore
                for param_name, param in block.named_parameters():  # type: ignore
                    param.requires_grad = True
                    full_name = f"vision_backbone.trunk.blocks.{i}.{param_name}"
                    self._finetuned_param_names.add(full_name)

            # Unfreeze neck convs if requested
            if self.finetune_neck_convs:
                # SAM3 neck has 4 convs (one per scale factor)
                for i in range(len(self.vision_backbone.convs)):  # type: ignore
                    conv = self.vision_backbone.convs[i]  # type: ignore
                    for param_name, param in conv.named_parameters():  # type: ignore
                        param.requires_grad = True
                        full_name = f"vision_backbone.convs.{i}.{param_name}"
                        self._finetuned_param_names.add(full_name)

            logger.info(
                f"SAM3 encoder fine-tuning: ViT blocks {start_block}-{num_blocks - 1}, "
                f"neck_convs={self.finetune_neck_convs} "
                f"({len(self._finetuned_param_names)} parameters)"
            )

    def get_finetuned_param_names(self) -> set[str]:
        """Return set of parameter names that are unfrozen for fine-tuning."""
        return self._finetuned_param_names.copy()

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input images to SAM3 required size (1008x1008).

        Args:
            images: (B, C, H, W) input images

        Returns:
            preprocessed_images: (B, C, 1008, 1008) preprocessed images
        """
        # Resize images to 1008x1008 if needed (SAM3 native resolution)
        if images.shape[-2] != 1008 or images.shape[-1] != 1008:
            return torch.nn.functional.interpolate(
                images, (1008, 1008), mode="bilinear", align_corners=False
            )

        return images

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Size]:
        """
        Forward pass through SAM3 image encoder.

        Args:
            images: (B, C, H, W) input images of any size

        Returns:
            vision_features: (B, 256, 36, 36) lowest resolution FPN features
                These are deep, semantic-rich features for the decoder.

            backbone_fpn: List of FPN features at multiple scales
                [(B, 256, 288, 288),  # scale=4.0: Highest resolution (4x upsampling)
                 (B, 256, 144, 144),  # scale=2.0: High resolution (2x upsampling)
                 (B, 256, 72, 72),    # scale=1.0: Medium resolution (no change)
                 (B, 256, 36, 36)]    # scale=0.5: Low resolution (2x downsampling)

            vision_pos_enc: List of positional encodings for each FPN level
                Same shapes as backbone_fpn, used for attention in decoder.

            original_shape: Original input image shape (B, C, H, W)
        """
        # Store original shape
        original_shape = images.shape

        # Preprocess images to 1008x1000 for SAM3
        preprocessed_images = self.preprocess_images(images)

        # Forward through SAM3 vision backbone
        # Sam3DualViTDetNeck.forward() returns:
        # - sam3_out: List of 4 feature maps at different scales
        # - sam3_pos: List of 4 positional encodings
        # - sam2_out, sam2_pos: Optional SAM2 features (None if add_sam2_neck=False)
        #
        # Note: The neck expects a single tensor input (not wrapped in list)
        # Note: inference_mode is applied via decorator at function level
        trunk_outputs = self.vision_backbone.trunk(preprocessed_images)  # type: ignore
        trunk_feature = trunk_outputs[-1]
        position_encodings = (
            self.position_encoding_0,
            self.position_encoding_1,
            self.position_encoding_2,
            self.position_encoding_3,
        )
        sam3_fpn = []
        sam3_pos_enc = []
        for conv, position in zip(
            self.vision_backbone.convs,  # type: ignore
            position_encodings,
        ):
            feature = conv(trunk_feature)
            sam3_fpn.append(feature)
            sam3_pos_enc.append(
                position.unsqueeze(0).expand(feature.shape[0], -1, -1, -1).to(dtype=feature.dtype)
            )

        # Extract outputs
        # sam3_fpn is a list of 4 feature maps:
        # [(B, 256, 288, 288), (B, 256, 144, 144), (B, 256, 72, 72), (B, 256, 36, 36)]
        # for 1008x1008 input (patch_size=14 → 72x72 grid, FPN scales [4.0, 2.0, 1.0, 0.5])

        # vision_features: Use the lowest resolution (last) feature map
        # This is (B, 256, 36, 36) for 1008x1008 input
        vision_features = sam3_fpn[-1]

        return vision_features, sam3_fpn, sam3_pos_enc, original_shape


def build_sam3_encoder(
    sam3_checkpoint: Path | None = None,
    device: str = "cuda",
    finetune: bool = False,
    finetune_last_n_blocks: int = 2,
    finetune_neck_convs: bool = True,
    enable_inst_interactivity: bool = False,
) -> SAM3Encoder:
    """
    Build SAM3 encoder from checkpoint or download pretrained weights.

    Args:
        sam3_checkpoint: Path to SAM3 checkpoint file. If None, downloads from HF.
        device: Device to load model on ('cuda' or 'cpu')
        finetune: Whether to enable fine-tuning
        finetune_last_n_blocks: Number of last ViT blocks to fine-tune
        finetune_neck_convs: Whether to fine-tune neck convolutions
        enable_inst_interactivity: Whether to enable SAM2-style interactivity

    Returns:
        SAM3Encoder: Wrapped SAM3 encoder

    Example:
        >>> encoder = build_sam3_encoder(
        ...     sam3_checkpoint="path/to/sam3.pt",
        ...     finetune=True,
        ...     finetune_last_n_blocks=2,
        ... )
        >>> images = torch.randn(2, 3, 1024, 1024).cuda()
        >>> vision_features, backbone_fpn, pos_enc, orig_shape = encoder(images)
        >>> print(vision_features.shape)  # (2, 256, 36, 36)
        >>> print([f.shape for f in backbone_fpn])  # [(2, 256, 288, 288), (2, 256, 144, 144), ...]
    """
    if build_sam3_image_model is None:
        raise ImportError(
            "SAM3 library not found. Install it from: https://github.com/facebookresearch/segment-anything-3"
        )

    if sam3_checkpoint is None:
        sam3_checkpoint = _pinned_sam3_checkpoint()
    else:
        sam3_checkpoint = verify_sam3_checkpoint(Path(sam3_checkpoint))

    # Build full SAM3 model
    sam3_model = build_sam3_image_model(
        checkpoint_path=sam3_checkpoint,
        device=device,
        eval_mode=False,  # Keep in training mode for fine-tuning
        enable_segmentation=True,
        enable_inst_interactivity=enable_inst_interactivity,
    )

    # Extract the backbone for encoder wrapper
    sam3_backbone = sam3_model.backbone  # SAM3VLBackbone

    # Create encoder wrapper
    encoder = SAM3Encoder(
        sam3_backbone=sam3_backbone,
        finetune=finetune,
        finetune_last_n_blocks=finetune_last_n_blocks,
        finetune_neck_convs=finetune_neck_convs,
    )

    # Configure freezing
    encoder.configure_freezing()

    logger.success("Built SAM3 encoder successfully")
    return encoder

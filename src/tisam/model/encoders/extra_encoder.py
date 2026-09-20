"""Extra encoder component for TIMM pathology foundation models.

`model.segmentor.Mask2FormerSegmentor` calls this module to turn full-resolution
tiles into a 4x4 grid of TIMM-compatible crops. Depending on the configured
feature mode, the pixel decoder consumes either the final spatial token map or
three depth-aligned maps for P3/P4/P5 dual-encoder fusion.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from prettyterm import get_logger
from timm.layers.mlp import SwiGLUPacked

logger = get_logger(__name__)

_EXTRA_ENCODER_GRID_SIZE = 4
_EXTRA_ENCODER_PATCH_GRID_SIZE = 16
_EXTRA_ENCODER_PATCH_TOKENS = _EXTRA_ENCODER_PATCH_GRID_SIZE**2

ExtraFeatureMode = Literal["final", "deep"]
ExtraFeaturePyramid = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
ExtraFeatures = torch.Tensor | ExtraFeaturePyramid


@dataclass(frozen=True)
class ExtraEncoderSpec:
    """Model-specific TIMM construction and token layout settings."""

    model_name: str
    model_kwargs: dict[str, Any]
    prefix_tokens: int
    use_forward_features: bool = False
    adaptation: ExtraEncoderAdaptation | None = None


@dataclass(frozen=True)
class ExtraEncoderAdaptation:
    """Pinned low-rank adaptation merged into an extra-encoder backbone."""

    repo_id: str
    filename: str
    revision: str
    sha256: str
    rank: int
    alpha: int


_UNI2_SEAL_ADAPTATION = ExtraEncoderAdaptation(
    repo_id="MahmoodLab/SEAL",
    filename="seal_univ2_vision.pth",
    revision="c20762e7a04551e21e62c083a9c443bdbc48cb89",
    sha256="e7ae736968c355ae02e565ee828f1ed6bd9ffe01b7dbc5a9166b3a55d746bd49",
    rank=8,
    alpha=8,
)


def _uni2_extra_encoder_spec(
    model_name: str,
    adaptation: ExtraEncoderAdaptation | None = None,
) -> ExtraEncoderSpec:
    """Return the shared UNI2-h architecture with an optional frozen adaptation."""
    return ExtraEncoderSpec(
        model_name=model_name,
        model_kwargs={
            "pretrained": True,
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        },
        prefix_tokens=9,
        use_forward_features=True,
        adaptation=adaptation,
    )


def _resolve_extra_encoder_spec(extra_encoder: str) -> ExtraEncoderSpec:
    """Return TIMM kwargs and token layout for supported extra encoders."""
    if extra_encoder == "hf-hub:paige-ai/Virchow2":
        return ExtraEncoderSpec(
            model_name=extra_encoder,
            model_kwargs={
                "pretrained": True,
                "mlp_layer": SwiGLUPacked,
                "act_layer": torch.nn.SiLU,
            },
            prefix_tokens=5,
            use_forward_features=False,
        )

    if extra_encoder == "hf-hub:MahmoodLab/UNI2-h":
        return _uni2_extra_encoder_spec(extra_encoder)

    if extra_encoder == "hf-hub:MahmoodLab/UNI2-SEAL":
        return _uni2_extra_encoder_spec(
            "hf-hub:MahmoodLab/UNI2-h",
            adaptation=_UNI2_SEAL_ADAPTATION,
        )

    raise ValueError(
        f"Unsupported extra encoder '{extra_encoder}'. "
        "Expected one of: 'hf-hub:paige-ai/Virchow2', "
        "'hf-hub:MahmoodLab/UNI2-h', 'hf-hub:MahmoodLab/UNI2-SEAL'."
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest used to verify a downloaded adaptation."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_seal_lora_adaptation(
    model: nn.Module,
    adaptation: ExtraEncoderAdaptation,
) -> None:
    """Merge the published UNI2-SEAL LoRA update into a TIMM UNI2-h backbone."""
    try:
        checkpoint_path = Path(
            hf_hub_download(
                repo_id=adaptation.repo_id,
                filename=adaptation.filename,
                revision=adaptation.revision,
            )
        )
    except Exception as error:
        raise RuntimeError(
            "Unable to download the gated UNI2-SEAL checkpoint. Request access to "
            f"https://huggingface.co/{adaptation.repo_id} and authenticate with "
            "`hf auth login` before using this encoder."
        ) from error
    observed_sha256 = _sha256(checkpoint_path)
    if observed_sha256 != adaptation.sha256:
        raise RuntimeError(
            f"UNI2-SEAL checkpoint digest mismatch: expected {adaptation.sha256}, "
            f"got {observed_sha256}."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError("UNI2-SEAL checkpoint does not contain a state dictionary.")

    prefix = "module.encoder.base_model.model."
    suffixes = {
        ".lora_A.default.weight": "A",
        ".lora_B.default.weight": "B",
    }
    lora_pairs: dict[str, dict[str, torch.Tensor]] = {}
    ignored_keys: set[str] = set()
    for key, value in state_dict.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError("UNI2-SEAL checkpoint contains a non-tensor state entry.")
        if key.startswith("module.projection_head.") or key.startswith("module.decoder."):
            ignored_keys.add(key)
            continue
        if not key.startswith(prefix):
            raise RuntimeError(f"Unexpected UNI2-SEAL checkpoint key: {key}")
        matched_suffix = next((suffix for suffix in suffixes if key.endswith(suffix)), None)
        if matched_suffix is None:
            raise RuntimeError(f"Unexpected UNI2-SEAL adaptation key: {key}")
        module_name = key[len(prefix) : -len(matched_suffix)]
        lora_pairs.setdefault(module_name, {})[suffixes[matched_suffix]] = value

    expected_modules = {
        f"blocks.{block}.{module_name}"
        for block in (21, 22, 23)
        for module_name in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
    }
    if set(lora_pairs) != expected_modules:
        missing = sorted(expected_modules - set(lora_pairs))
        unexpected = sorted(set(lora_pairs) - expected_modules)
        raise RuntimeError(
            f"UNI2-SEAL adaptation coverage mismatch; missing={missing}, unexpected={unexpected}."
        )
    if len(ignored_keys) != 6:
        raise RuntimeError(
            "UNI2-SEAL checkpoint must contain exactly the published projection and decoder "
            f"entries; found {len(ignored_keys)}."
        )

    scale = adaptation.alpha / adaptation.rank
    with torch.no_grad():
        for module_name, pair in lora_pairs.items():
            if set(pair) != {"A", "B"}:
                raise RuntimeError(f"Incomplete UNI2-SEAL LoRA pair for {module_name}.")
            target = model.get_submodule(module_name)
            if not isinstance(target, nn.Linear):
                raise RuntimeError(f"UNI2-SEAL target {module_name} is not a linear layer.")
            matrix_a = pair["A"]
            matrix_b = pair["B"]
            if matrix_a.shape[0] != adaptation.rank or matrix_b.shape[1] != adaptation.rank:
                raise RuntimeError(f"UNI2-SEAL rank mismatch for {module_name}.")
            delta = torch.matmul(matrix_b, matrix_a) * scale
            if delta.shape != target.weight.shape:
                raise RuntimeError(
                    f"UNI2-SEAL shape mismatch for {module_name}: "
                    f"expected {tuple(target.weight.shape)}, got {tuple(delta.shape)}."
                )
            target.weight.add_(delta.to(device=target.weight.device, dtype=target.weight.dtype))

    logger.success(
        f"Merged UNI2-SEAL adaptation {adaptation.filename} at {adaptation.revision} "
        f"(sha256={observed_sha256})"
    )


class ExtraEncoder(nn.Module):
    """
    TIMM extra encoder component.
    Handles patch cropping from images (HxW divisible by 4),
    interpolation to 224x224, and feature extraction.
    """

    def __init__(
        self,
        extra_encoder: str,
        extra_shape: tuple[int, int] = (224, 224),
        extra_embed_dim: int = 1280,
        finetune: bool = False,
        finetune_last_n_blocks: int = 2,
        feature_mode: ExtraFeatureMode = "final",
    ):
        super().__init__()

        if feature_mode not in {"final", "deep"}:
            raise ValueError(f"Unsupported extra feature mode: {feature_mode}")

        # Store configuration
        self.extra_shape = extra_shape
        self.extra_embed_dim = extra_embed_dim
        self.finetune = finetune
        self.finetune_last_n_blocks = finetune_last_n_blocks
        self.feature_mode = feature_mode
        self.encoder_spec = _resolve_extra_encoder_spec(extra_encoder)
        self.prefix_tokens = self.encoder_spec.prefix_tokens
        self.use_forward_features = self.encoder_spec.use_forward_features

        expected_embed_dim = cast(int, self.encoder_spec.model_kwargs.get("embed_dim", 1280))
        if expected_embed_dim != self.extra_embed_dim:
            raise ValueError(
                f"extra_embed_dim={self.extra_embed_dim} does not match "
                f"{extra_encoder} embed_dim={expected_embed_dim}"
            )

        self.extra_model = timm.create_model(
            model_name=self.encoder_spec.model_name,
            **self.encoder_spec.model_kwargs,
        )
        if self.encoder_spec.adaptation is not None:
            _merge_seal_lora_adaptation(self.extra_model, self.encoder_spec.adaptation)
        self.extra_model.eval()
        if self.feature_mode == "deep" and not hasattr(self.extra_model, "forward_intermediates"):
            raise ValueError(
                f"Extra encoder '{extra_encoder}' does not expose intermediate features."
            )
        self.deep_feature_indices = (
            self._resolve_deep_feature_indices() if self.feature_mode == "deep" else None
        )

        # Track fine-tuned parameter names for optimizer param groups
        self._finetuned_param_names: set[str] = set()

        deep_feature_blocks = (
            tuple(index + 1 for index in self.deep_feature_indices)
            if self.deep_feature_indices
            else None
        )
        logger.success(
            f"ExtraEncoder initialized with {extra_encoder}, embed_dim={extra_embed_dim}, "
            f"finetune={finetune}, finetune_last_n_blocks={finetune_last_n_blocks}, "
            f"feature_mode={feature_mode}, deep_feature_blocks="
            f"{deep_feature_blocks}"
        )

    def _resolve_deep_feature_indices(self) -> tuple[int, int, int]:
        """Select one-third, two-thirds, and final block outputs for `forward`."""
        blocks = cast(nn.ModuleList, self.extra_model.blocks)
        num_blocks = len(blocks)
        if num_blocks < 3:
            raise ValueError(
                "Deep extra-feature fusion requires at least three transformer blocks."
            )
        return (num_blocks // 3 - 1, (2 * num_blocks) // 3 - 1, num_blocks - 1)

    def configure_freezing(self):
        """Configure parameter freezing based on fine-tuning settings."""
        if not self.finetune:
            # Freeze all parameters
            for param in self.extra_model.parameters():
                param.requires_grad = False
            logger.info("Froze all extra encoder parameters")
        else:
            # Get total number of blocks
            num_blocks = len(self.extra_model.blocks)  # type: ignore
            start_block = max(0, num_blocks - self.finetune_last_n_blocks)

            # First freeze all parameters
            for param in self.extra_model.parameters():
                param.requires_grad = False

            # Then unfreeze the last n blocks
            for i in range(start_block, num_blocks):
                block = cast(nn.Module, self.extra_model.blocks[i])  # type: ignore
                for param_name, param in block.named_parameters():
                    param.requires_grad = True
                    full_name = f"extra_model.blocks.{i}.{param_name}"
                    self._finetuned_param_names.add(full_name)

            logger.info(
                f"Fine-tuning enabled: unfrozen blocks {start_block}-{num_blocks - 1} "
                f"({len(self._finetuned_param_names)} parameters)"
            )

    def get_finetuned_param_names(self) -> set[str]:
        """Return set of parameter names that are unfrozen for fine-tuning."""
        return self._finetuned_param_names.copy()

    def _extract_final_patch_tokens(self, interpolated: torch.Tensor) -> torch.Tensor:
        """Return final patch tokens for the legacy single-depth path."""
        if self.use_forward_features:
            forward_features = cast(
                Callable[[torch.Tensor], torch.Tensor], self.extra_model.forward_features
            )
            output_tokens = forward_features(interpolated)
        else:
            output_tokens = self.extra_model(interpolated)

        if output_tokens.ndim != 3:
            raise ValueError(
                "Expected extra encoder output with shape [n, tokens, channels], "
                f"got {output_tokens.shape}"
            )
        return output_tokens[:, self.prefix_tokens :, :]

    def _extract_deep_patch_tokens(self, interpolated: torch.Tensor) -> ExtraFeaturePyramid:
        """Return normalized shallow, middle, and final TIMM block patch tokens."""
        if self.deep_feature_indices is None:
            raise RuntimeError("Deep feature indices were not initialized.")

        forward_intermediates = cast(
            Callable[..., list[torch.Tensor]], self.extra_model.forward_intermediates
        )
        outputs = forward_intermediates(
            interpolated,
            indices=list(self.deep_feature_indices),
            norm=True,
            output_fmt="NLC",
            intermediates_only=True,
        )
        if len(outputs) != 3:
            raise ValueError(f"Expected three intermediate extra features, got {len(outputs)}")
        return outputs[0], outputs[1], outputs[2]

    def _stitch_patch_tokens(self, patch_tokens: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Stitch per-crop patch tokens into one spatial map for the pixel decoder."""
        expected_crops = batch_size * _EXTRA_ENCODER_GRID_SIZE**2
        if patch_tokens.ndim != 3 or patch_tokens.shape[0] != expected_crops:
            raise ValueError(
                f"Expected patch tokens for {expected_crops} crops, got {patch_tokens.shape}"
            )
        if patch_tokens.shape[1] != _EXTRA_ENCODER_PATCH_TOKENS:
            raise ValueError(
                f"Expected {_EXTRA_ENCODER_PATCH_TOKENS} patch tokens, got {patch_tokens.shape[1]}"
            )
        if patch_tokens.shape[2] != self.extra_embed_dim:
            raise ValueError(
                f"Expected patch embedding dim {self.extra_embed_dim}, got {patch_tokens.shape[2]}"
            )

        patch_tokens_2d = patch_tokens.reshape(
            batch_size,
            _EXTRA_ENCODER_GRID_SIZE,
            _EXTRA_ENCODER_GRID_SIZE,
            _EXTRA_ENCODER_PATCH_GRID_SIZE,
            _EXTRA_ENCODER_PATCH_GRID_SIZE,
            self.extra_embed_dim,
        )
        spatial_size = _EXTRA_ENCODER_GRID_SIZE * _EXTRA_ENCODER_PATCH_GRID_SIZE
        patch_tokens_spatial = patch_tokens_2d.permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size,
            self.extra_embed_dim,
            spatial_size,
            spatial_size,
        )
        return patch_tokens_spatial

    def forward(self, images: torch.Tensor) -> ExtraFeatures:
        """Run the configured TIMM encoder on 4x4 tile crops.

        `Mask2FormerSegmentor.forward` calls this before passing the spatial
        token map to `PixelDecoder` for extra-feature fusion. The tile is resized
        once to the whole 4x4 crop grid, then reshaped into TIMM inputs so the
        preprocessing path avoids a larger 16x-crop interpolation call.

        Args:
            images: (b, 3, h, w) input images (h, w must be divisible by 4)

        Returns:
            One final spatial map in `final` mode, or a P3/P4/P5-ordered tuple
            of shallow, middle, and final maps in `deep` mode. Every map has
            shape `(b, extra_embed_dim, 64, 64)`.
        """
        b, c, h, w = images.shape
        grid_h = self.extra_shape[0] * _EXTRA_ENCODER_GRID_SIZE
        grid_w = self.extra_shape[1] * _EXTRA_ENCODER_GRID_SIZE

        resized_grid = torch.nn.functional.interpolate(
            images,
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False,
        )
        interpolated = resized_grid.reshape(
            b,
            c,
            _EXTRA_ENCODER_GRID_SIZE,
            self.extra_shape[0],
            _EXTRA_ENCODER_GRID_SIZE,
            self.extra_shape[1],
        )
        interpolated = interpolated.permute(0, 2, 4, 1, 3, 5).reshape(
            -1,
            c,
            self.extra_shape[0],
            self.extra_shape[1],
        )

        grad_context = torch.set_grad_enabled(
            self.finetune and self.training and torch.is_grad_enabled()
        )
        with grad_context:
            if self.feature_mode == "deep":
                p3_tokens, p4_tokens, p5_tokens = self._extract_deep_patch_tokens(interpolated)
                return (
                    self._stitch_patch_tokens(p3_tokens, b),
                    self._stitch_patch_tokens(p4_tokens, b),
                    self._stitch_patch_tokens(p5_tokens, b),
                )
            patch_tokens = self._extract_final_patch_tokens(interpolated)
            return self._stitch_patch_tokens(patch_tokens, b)

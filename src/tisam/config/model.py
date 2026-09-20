"""Architecture-only configuration consumed by TiSAM and checkpoint loaders."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    """Describe a dual-encoder model independently of datasets and training."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    num_classes: int = Field(gt=0)
    total_queries: int | None = None
    d_model: int = 256
    num_layers: int = 6
    n_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    upsampling_stages: int = 3
    finetune: bool = False
    finetune_last_n_blocks: int = 2
    finetune_neck_convs: bool = True
    output_hw: tuple[int, int] = (1008, 1008)
    extra_encoder: Literal[
        "hf-hub:MahmoodLab/UNI2-h", "hf-hub:paige-ai/Virchow2", "hf-hub:MahmoodLab/UNI2-SEAL"
    ] = Field()
    extra_shape: tuple[int, int] = (224, 224)
    extra_embed_dim: int | None = None
    extra_finetune: bool = False
    extra_finetune_last_n_blocks: int = 2
    extra_feature_mode: Literal["final", "deep"] = "final"
    pos_embed_mode: Literal["none", "rope", "learned"] = "none"
    learned_pos_embed_init: Literal["normal"] = "normal"
    learned_pos_embed_std: float = 0.02
    learned_image_pos_scope: Literal["per_level"] = "per_level"
    learned_query_pos: bool = True
    learned_image_pos: bool = True
    use_cross_attn: bool = False
    hard_attn: bool = False
    use_sdpa_attn: bool = True
    sam_proj: bool = False
    early_interpolation: bool = True

    @model_validator(mode="after")
    def validate_geometry(self) -> ModelConfig:
        """Resolve query count and backbone width before constructing modules."""
        width = 1280 if self.extra_encoder == "hf-hub:paige-ai/Virchow2" else 1536
        if self.extra_embed_dim is not None and self.extra_embed_dim != width:
            raise ValueError("extra_embed_dim does not match the selected foundation encoder")
        object.__setattr__(self, "extra_embed_dim", width)
        if self.total_queries is None:
            object.__setattr__(self, "total_queries", self.num_classes + 1)
        for name in ("total_queries", "d_model", "num_layers", "n_heads", "dim_feedforward"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if any(v <= 0 for v in self.output_hw) or self.extra_shape != (224, 224):
            raise ValueError(
                "output_hw must be positive; foundation encoders require extra_shape=(224,224)"
            )
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if self.learned_pos_embed_std <= 0:
            raise ValueError("learned_pos_embed_std must be positive")
        if self.pos_embed_mode == "learned" and not (
            self.learned_query_pos or self.learned_image_pos
        ):
            raise ValueError("Learned PE requires query or image positions")
        return self

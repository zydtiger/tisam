"""Training and evaluation settings composed by Python and CLI entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .data import DataConfig
from .model import ModelConfig


class TrainConfig(BaseModel):
    """Compose architecture and datasets with a single-process training policy."""

    model_config = ConfigDict(extra="forbid")
    model: ModelConfig
    data: DataConfig
    out_dir: Path = Path("runs")
    name: str = "tisam"
    device: str = "cuda"
    sam3_checkpoint: Optional[Path] = None
    seed: int = Field(default=42, ge=0)
    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=1, gt=0)
    num_workers: int = Field(default=0, ge=0)
    persistent_workers: bool = False
    accumulation_steps: int = Field(default=16, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-2, ge=0)
    warmup_ratio: float = Field(default=0.05, ge=0, lt=1)
    image_lr_ratio: float = Field(default=0.1, gt=0)
    extra_lr_ratio: float = Field(default=0.1, gt=0)
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    compile: bool = False
    loss: Literal["median_frequency_weighted_ce", "focal_dice", "tempered_weighted_ce_dice_bg"] = (
        "median_frequency_weighted_ce"
    )
    focal_coefficient: float = Field(default=1, ge=0)
    dice_coefficient: float = Field(default=1, ge=0)
    weighted_ce_coefficient: float = Field(default=1, gt=0)
    tempered_weight_power: float = Field(default=0.5, gt=0)
    tissue_background_coefficient: float = Field(default=0.5, ge=0)
    early_stopping_monitor: Literal["total_loss", "mean_fg_iou"] = "total_loss"
    early_stopping_mode: Literal["min", "max"] = "min"
    patience: int = Field(default=10, gt=0)
    min_delta: float = Field(default=0.001, ge=0)
    checkpoint_mode: Literal["all", "latest"] = "latest"

    @model_validator(mode="after")
    def validate_contract(self) -> TrainConfig:
        """Check model/dataset agreement before constructing any expensive model."""
        if len(self.data.class_names) != self.model.num_classes + 1:
            raise ValueError("model.num_classes excludes void and must match data.class_names")
        if Path(self.name).name != self.name or self.name in (".", "..", ""):
            raise ValueError("name must be a single directory name")
        if self.loss == "focal_dice" and self.focal_coefficient + self.dice_coefficient == 0:
            raise ValueError("At least one focal/Dice coefficient must be positive")
        if any(s.metadata.train_void_canvas_policy == "enforce_black" for s in self.data.train):
            if (self.early_stopping_monitor, self.early_stopping_mode) != ("mean_fg_iou", "max"):
                raise ValueError("enforce_black requires mean_fg_iou/max checkpoint selection")
        return self


def load_config(path: str | Path) -> TrainConfig:
    """Read YAML and resolve input paths relative to its directory."""
    path = Path(path).resolve()
    config = TrainConfig.model_validate(yaml.safe_load(path.read_text()))
    for split in (config.data.train, config.data.validation, config.data.test):
        for source in split:
            if not source.path.is_absolute():
                source.path = (path.parent / source.path).resolve()
    if config.sam3_checkpoint is not None and not config.sam3_checkpoint.is_absolute():
        config.sam3_checkpoint = (path.parent / config.sam3_checkpoint).resolve()
    # Outputs remain relative to the invoking working directory, default ./runs.
    return config

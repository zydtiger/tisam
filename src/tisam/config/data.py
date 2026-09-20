"""Dataset configuration and raw-label policies consumed by data loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class DatasetMetadata(BaseModel):
    """Map raw mask IDs to canonical class names before supervision and metrics."""

    model_config = ConfigDict(extra="forbid")
    name: str = "default"
    label_to_class_name: list[str]
    supervision_ignored_raw_labels: list[StrictInt] = Field(default_factory=list)
    metric_ignored_raw_labels: list[StrictInt] = Field(default_factory=lambda: [0])
    train_void_canvas_policy: Literal["preserve", "enforce_black"] = "preserve"

    @model_validator(mode="after")
    def validate_labels(self) -> DatasetMetadata:
        """Validate IDs before the tile dataset remaps them."""
        if not self.label_to_class_name:
            raise ValueError("label_to_class_name cannot be empty")
        for ids in (self.supervision_ignored_raw_labels, self.metric_ignored_raw_labels):
            if len(ids) != len(set(ids)) or any(
                i < 0 or i >= len(self.label_to_class_name) for i in ids
            ):
                raise ValueError("Ignored raw IDs must be unique and within the label mapping")
        if not set(self.supervision_ignored_raw_labels) <= set(self.metric_ignored_raw_labels):
            raise ValueError("Supervision-ignored IDs must also be metric-ignored")
        return self


@dataclass(frozen=True)
class ResolvedDataset:
    """Pair a data root with its raw-label contract for TileDataset."""

    path: Path
    metadata: DatasetMetadata


class DatasetSource(BaseModel):
    """One paired image/label directory and its raw-label metadata."""

    model_config = ConfigDict(extra="forbid")
    path: Path
    metadata: DatasetMetadata


class DataConfig(BaseModel):
    """User-owned tile roots, canonical classes and preprocessing parameters."""

    model_config = ConfigDict(extra="forbid")
    class_names: list[str]
    train: list[DatasetSource] = Field(default_factory=list)
    validation: list[DatasetSource] = Field(default_factory=list)
    test: list[DatasetSource] = Field(default_factory=list)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    crop_scale: tuple[float, float] = (0.5, 1.0)
    color_jitter_p: float = Field(default=0.5, ge=0, le=1)
    synthetic_background_fraction: float = Field(default=0, ge=0, lt=1)
    synthetic_background_seed: int = 0

    @model_validator(mode="after")
    def validate_sources(self) -> DataConfig:
        """Require one canonical void plus exact, unique tissue identifiers."""
        if len(self.class_names) < 2 or self.class_names[0] != "void":
            raise ValueError("class_names must start with void and contain tissue classes")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique")
        if any(v <= 0 for v in self.std):
            raise ValueError("Normalization standard deviations must be positive")
        if not 0 < self.crop_scale[0] <= self.crop_scale[1] <= 1:
            raise ValueError("crop_scale must satisfy 0 < min <= max <= 1")
        for split in (self.train, self.validation, self.test):
            if len({s.metadata.name for s in split}) != len(split):
                raise ValueError("Dataset names must be unique within each split")
            for source in split:
                if not set(source.metadata.label_to_class_name) <= set(self.class_names):
                    raise ValueError("Raw label mapping references an unknown canonical class")
        return self

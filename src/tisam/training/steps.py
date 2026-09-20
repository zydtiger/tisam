"""Dense TiSAM forward/loss steps used by the Mammoth trainer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from mammoth.torch import StepContext, StepFunction, StepOutput


@dataclass(frozen=True, slots=True)
class ArchitectureSteps:
    """Pair TiSAM's model-specific train and validation callables for Mammoth."""

    train_step: StepFunction
    validation_step: StepFunction


@dataclass(frozen=True, slots=True)
class DenseSegmentationSteps:
    """Apply one configured dense segmentation loss to model logits."""

    loss_fn: torch.nn.Module

    def train_step(
        self,
        model: torch.nn.Module,
        batch: Any,
        context: StepContext,
    ) -> StepOutput:
        """Return dense-logit losses for one training batch."""
        del context
        images, target_masks = batch[:2]
        losses = validated_losses(self.loss_fn(model(images), target_masks), "training")
        return training_output(losses)

    def validation_step(
        self,
        model: torch.nn.Module,
        batch: Any,
        context: StepContext,
    ) -> StepOutput:
        """Return dense losses and IoU updates for one validation batch."""
        del context
        images, target_masks = batch[:2]
        pred_logits = model(images)
        losses = validated_losses(self.loss_fn(pred_logits, target_masks), "validation")
        return validation_output(losses, pred_logits, target_masks, batch)

    def bundle(self) -> ArchitectureSteps:
        """Expose this provider through Mammoth's two callable contracts."""
        return ArchitectureSteps(self.train_step, self.validation_step)


def validated_losses(value: Any, phase: str) -> dict[str, torch.Tensor]:
    """Validate TiSAM's scalar loss-dictionary contract."""
    if not isinstance(value, dict) or any(
        not isinstance(name, str) or not isinstance(loss, torch.Tensor)
        for name, loss in value.items()
    ):
        raise TypeError(f"{phase} losses must be a string-to-tensor dictionary")
    losses = cast(dict[str, torch.Tensor], value)
    if "total_loss" not in losses:
        available = ", ".join(sorted(losses)) if losses else "<none>"
        raise KeyError(f"{phase} losses must include 'total_loss'; available keys: {available}")
    for name, loss in losses.items():
        if loss.ndim != 0:
            raise ValueError(
                f"{phase} loss '{name}' must be a scalar tensor; got shape {tuple(loss.shape)}"
            )
    return losses


def training_output(losses: Mapping[str, torch.Tensor]) -> StepOutput:
    """Translate TiSAM training losses into Mammoth's generic step output."""
    return StepOutput(
        loss=losses["total_loss"],
        metrics={name: value for name, value in losses.items() if name != "total_loss"},
    )


def validation_output(
    losses: Mapping[str, torch.Tensor],
    pred_logits: torch.Tensor,
    target_masks: torch.Tensor,
    batch: Any,
) -> StepOutput:
    """Translate segmentation losses and predictions into Mammoth metric updates."""
    metric_valid = batch[2] if len(batch) == 3 else torch.ones_like(target_masks, dtype=torch.bool)
    # Supervision-ignored pixels reported class 0 under the former all-zero
    # one-hot row; the metric-ignore policy in `metric_valid` still excludes
    # whichever pixels the dataset marks as unevaluated.
    target_indices = target_masks.clamp_min(0)
    pred_indices = align_prediction_indices(pred_logits.argmax(dim=1), target_indices)
    return StepOutput(
        metrics=losses,
        metric_updates={
            "segmentation_iou": (
                pred_indices,
                target_indices,
                metric_valid.to(device=target_indices.device, non_blocking=True),
            )
        },
    )


def align_prediction_indices(
    pred_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> torch.Tensor:
    """Resize predicted class indices to the target spatial shape when needed."""
    if pred_indices.shape[-2:] == target_indices.shape[-2:]:
        return pred_indices
    return (
        torch.nn.functional.interpolate(
            pred_indices.unsqueeze(1).float(),
            size=target_indices.shape[-2:],
            mode="nearest-exact",
        )
        .squeeze(1)
        .long()
    )

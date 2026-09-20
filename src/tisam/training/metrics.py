"""Additive segmentation counters used by Mammoth validation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


class SegmentationIOUMetric:
    """Expose TiSAM semantic IoU counters through Mammoth's additive metric API."""

    def __init__(
        self,
        *,
        num_classes: int,
        device: str,
        class_names: list[str],
        evaluated_class_ids: list[int],
        class_groups: Mapping[str, list[int]] | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.class_names = class_names
        self.evaluated_class_ids = evaluated_class_ids
        self.class_groups = dict(class_groups or {})
        self.reset()

    def reset(self) -> None:
        """Clear device-resident intersection, union, and validity counters."""
        self.intersection = torch.zeros(
            self.num_classes,
            dtype=torch.long,
            device=self.device,
        )
        self.union = torch.zeros_like(self.intersection)
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes) if self.class_groups else (0, 0),
            dtype=torch.long,
            device=self.device,
        )
        self.evaluated_pixel_count = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device,
        )

    def update(self, value: Any) -> None:
        """Consume one TiSAM prediction, target, and validity-mask tuple."""
        if not isinstance(value, tuple) or len(value) != 3:
            raise TypeError("TiSAM IoU updates must be (prediction, target, valid) tuples")
        pred, target, valid = value
        if not all(isinstance(item, torch.Tensor) for item in value):
            raise TypeError("TiSAM IoU updates must contain tensors")
        self.evaluated_pixel_count += valid.sum()
        pred = pred[valid]
        target = target[valid]
        if target.numel() == 0:
            return
        pair_counts = torch.bincount(
            target * self.num_classes + pred,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)
        if self.class_groups:
            self.confusion += pair_counts
        intersection = pair_counts.diagonal()
        self.intersection += intersection
        self.union += pair_counts.sum(dim=0) + pair_counts.sum(dim=1) - intersection

    def state_tensors(self) -> Mapping[str, torch.Tensor]:
        """Return additive counters for Mammoth's all-rank reduction."""
        counters = {
            "intersection": self.intersection,
            "union": self.union,
            "evaluated_pixel_count": self.evaluated_pixel_count,
        }
        if self.class_groups:
            counters["confusion"] = self.confusion
        return counters

    def compute(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, float | torch.Tensor]:
        """Convert globally reduced counters into TiSAM metric names."""
        if int(state["evaluated_pixel_count"].item()) == 0:
            raise ValueError(
                "Validation contains no evaluated pixels after applying the configured "
                "metric-ignore policy."
            )
        intersection = state["intersection"]
        union = state["union"]
        per_class: dict[int, float] = {}
        metrics: dict[str, float] = {}
        for class_id in range(self.num_classes):
            value = (
                float("nan")
                if int(union[class_id].item()) == 0
                else float((intersection[class_id].float() / union[class_id].float()).item())
            )
            per_class[class_id] = value
            if class_id in self.evaluated_class_ids and not math.isnan(value):
                class_name = (
                    self.class_names[class_id]
                    if class_id < len(self.class_names)
                    else f"class_{class_id}"
                )
                metrics[f"iou_{class_name}"] = value
        evaluated = set(self.evaluated_class_ids)
        background = per_class.get(0, 0.0) if 0 in evaluated else 0.0
        metrics["bg_iou"] = 0.0 if math.isnan(background) else background
        foreground = [
            value
            for class_id, value in per_class.items()
            if class_id > 0 and class_id in evaluated and not math.isnan(value)
        ]
        metrics["mean_fg_iou"] = sum(foreground) / len(foreground) if foreground else 0.0
        for group, ids in self.class_groups.items():
            # Disjoint ontology class IDs identify each dataset's evaluated
            # ground-truth pixels. Keep all prediction columns, so predictions
            # outside that dataset's classes remain errors.
            counts = state["confusion"][ids].double()
            support = counts.sum(dim=1)
            total = support.sum()
            if total.item() == 0:
                continue
            tp = counts[:, ids].diagonal()
            predicted = counts[:, ids].sum(dim=0)
            present = support > 0
            precision = tp / predicted.clamp_min(1)
            recall = tp / support.clamp_min(1)
            f1 = 2 * tp / (support + predicted).clamp_min(1)
            metrics[f"{group}_accuracy"] = float((tp.sum() / total).item())
            for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
                metrics[f"{group}_{name}"] = float(values[present].mean().item())
        return metrics

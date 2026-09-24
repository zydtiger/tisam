"""Tile-based validation/test metrics using the same raw-label contract as training."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from tisam.config.data import ResolvedDataset
from tisam.config.run import TrainConfig
from tisam.data.tile_dataset import TileDataset


def evaluate(model, config: TrainConfig, *, split: str = "validation") -> dict:
    """Return confusion counts and foreground Accuracy/Precision/Recall/F1."""
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    sources = getattr(config.data, split)
    if not sources:
        raise ValueError(f"No {split} sources configured")
    dataset = TileDataset(
        datasets=[ResolvedDataset(s.path, s.metadata) for s in sources],
        dataset_mean=config.data.mean,
        dataset_std=config.data.std,
        num_classes=config.model.num_classes,
        dataset_class_names=config.data.class_names,
        target_size=config.model.input_hw,
        data_type="val",
        return_metric_validity=True,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    classes = len(config.data.class_names)
    counts = torch.zeros((classes, classes), dtype=torch.int64)
    device = next(model.parameters()).device
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            for images, target, valid in loader:
                logits = model(images.to(device))
                predictions = logits.argmax(1).cpu()
                if predictions.shape[-2:] != target.shape[-2:]:
                    predictions = torch.nn.functional.interpolate(
                        predictions[:, None].float(), size=target.shape[-2:], mode="nearest-exact"
                    )[:, 0].long()
                valid &= target >= 0
                counts += torch.bincount(
                    target[valid] * classes + predictions[valid], minlength=classes * classes
                ).reshape(classes, classes)
    finally:
        model.train(was_training)
    if not counts.sum():
        raise ValueError("Evaluation contains no valid pixels")
    tp = counts.diagonal().double()
    support = counts.sum(1).double()
    predicted = counts.sum(0).double()
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * tp / (support + predicted).clamp_min(1)
    present = support[1:] > 0
    if not present.any():
        raise ValueError("Evaluation contains no foreground support")
    return {
        "split": split.title(),
        "class_names": config.data.class_names,
        "confusion_matrix": counts.tolist(),
        "accuracy": float(tp[1:].sum() / support[1:].sum()),
        "precision": float(precision[1:][present].mean()),
        "recall": float(recall[1:][present].mean()),
        "f1": float(f1[1:][present].mean()),
    }

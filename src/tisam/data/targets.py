"""Integer target masks shared by TiSAM datasets, losses and metrics."""

from __future__ import annotations

from collections.abc import Sequence

import torch

SUPERVISION_IGNORE_INDEX = -1
"""Sentinel class id marking pixels excluded from supervision.

`TileDataset.__getitem__` writes this value wherever a dataset's
`supervision_ignored_raw_labels` policy applies. It replaces the former
all-zero one-hot row, so every consumer must treat it as "no supervision"
rather than as a class.
"""


def target_supervision_validity(target_indices: torch.Tensor) -> torch.Tensor:
    """Return the pixels that carry supervision in a class-index target."""
    return target_indices != SUPERVISION_IGNORE_INDEX


def resize_target_indices(
    target_indices: torch.Tensor,
    size: Sequence[int],
) -> torch.Tensor:
    """Resize ``BHW`` int64 class indices with nearest-exact interpolation."""
    if target_indices.ndim != 3:
        raise ValueError(
            f"Class-index targets must have shape (B, H, W), got {tuple(target_indices.shape)}."
        )
    if target_indices.dtype != torch.long:
        raise TypeError(f"Class-index targets must use torch.int64, got {target_indices.dtype}.")
    if len(size) != 2:
        raise ValueError(f"Target size must contain height and width, got {tuple(size)}.")
    target_size = (int(size[0]), int(size[1]))
    if target_indices.shape[-2:] == target_size:
        return target_indices
    return (
        torch.nn.functional.interpolate(
            target_indices.unsqueeze(1).float(),
            size=target_size,
            mode="nearest-exact",
        )
        .squeeze(1)
        .long()
    )

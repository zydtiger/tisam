"""Raw label validity and canonical remapping shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

__all__ = [
    "build_raw_to_global_class_map",
    "normalize_policy_raw_mask",
    "raw_label_validity",
    "remap_mask_to_global_classes",
]


def raw_label_validity(
    labels: np.ndarray,
    ignored_raw_labels: Sequence[int],
) -> np.ndarray:
    """Return a mask that is True where `labels` matches no ignored raw label.

    Called by `tisam.data.tile_dataset` for both the supervision and metric
    policies, and by `tisam.cli.eval.compute_metrics`,
    `tisam.cli.eval.inference_scope`, and `tisam.data.metrics` for the metric one.
    Each caller owns its out-of-range policy, so this never range-checks or remaps,
    and it validates nothing because `DatasetMetadata` already did.

    Compares at the labels' stored dtype, leaves the input untouched, and always
    returns an independent writable array.
    """
    if not ignored_raw_labels:
        return np.ones(labels.shape, dtype=bool)
    valid = labels != ignored_raw_labels[0]
    for ignored in ignored_raw_labels[1:]:
        valid &= labels != ignored
    return valid


def build_raw_to_global_class_map(
    label_to_class_name: Sequence[str],
    global_class_to_idx: Mapping[str, int],
) -> np.ndarray:
    """Build a dense raw-label -> canonical-class lookup table for one dataset."""
    return np.array(
        [global_class_to_idx[class_name] for class_name in label_to_class_name],
        dtype=np.int64,
    )


def remap_mask_to_global_classes(
    mask: np.ndarray,
    raw_to_global_class_idx: np.ndarray,
) -> np.ndarray:
    """Convert dataset-specific raw label ids into the canonical global class space."""
    raw_mask = mask.astype(np.int64, copy=False)
    invalid_mask = (raw_mask < 0) | (raw_mask >= len(raw_to_global_class_idx))
    if np.any(invalid_mask):  # NOTE: set invalid mask to void class (0)
        raw_mask = raw_mask.copy()
        raw_mask[invalid_mask] = 0

    return raw_to_global_class_idx[raw_mask]


def normalize_policy_raw_mask(
    mask: np.ndarray,
    raw_label_count: int,
) -> np.ndarray:
    """Map out-of-range raw IDs to 0, matching `remap_mask_to_global_classes`.

    Preserves the input dtype, and returns the caller's array unchanged when
    nothing is out of range, so treat the result as read-only.
    """
    invalid_mask = (mask < 0) | (mask >= raw_label_count)
    if not np.any(invalid_mask):
        return mask
    normalized = mask.copy()
    normalized[invalid_mask] = 0
    return normalized

"""Tests for tisam_torch training losses.

The loss module is consumed by training loops in tisam.training and
expects model logits plus int64 class-index masks.
"""

from __future__ import annotations

import torch

from tisam.training.losses import (
    DiceLoss,
    FocalLoss,
    MedianFrequencyWeightedCrossEntropyLoss,
    SegmentorLoss,
)


def _legacy_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Build the former dense target representation for equivalence checks."""
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).permute(0, 3, 1, 2).float()


def test_dice_loss_is_zero_for_perfect_foreground_overlap() -> None:
    """DiceLoss should reward exact foreground probability/target overlap."""
    pred_probs = torch.zeros(1, 3, 4, 4)
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    pred_probs[:, 1, 1:3, 1:3] = 1.0
    target[:, 1:3, 1:3] = 1

    loss = DiceLoss(include_background=False)(pred_probs, target)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_dice_loss_excludes_background_when_configured() -> None:
    """Background-only targets should not dominate foreground-only dice."""
    pred_probs = torch.zeros(1, 2, 2, 2)
    target = torch.zeros(1, 2, 2, dtype=torch.long)
    pred_probs[:, 0] = 1.0

    foreground_loss = DiceLoss(include_background=False)(pred_probs, target)
    background_loss = DiceLoss(include_background=True)(pred_probs, target)

    torch.testing.assert_close(background_loss, torch.tensor(0.0))
    torch.testing.assert_close(foreground_loss, torch.tensor(0.0))


def test_sparse_dice_matches_legacy_one_hot_value_and_gradient() -> None:
    """Sparse gather/scatter Dice should preserve dense one-hot supervision."""
    torch.manual_seed(7)
    target = torch.randint(0, 4, (2, 5, 6))
    sparse_logits = torch.randn(2, 4, 5, 6, requires_grad=True)
    legacy_logits = sparse_logits.detach().clone().requires_grad_(True)

    sparse_probs = torch.softmax(sparse_logits, dim=1)
    sparse_loss = DiceLoss(include_background=False)(sparse_probs, target)
    sparse_loss.backward()

    legacy_probs = torch.softmax(legacy_logits, dim=1)
    legacy_target = _legacy_one_hot(target, num_classes=4)
    legacy_intersection = (legacy_probs[:, 1:] * legacy_target[:, 1:]).sum(dim=(2, 3))
    legacy_denominator = legacy_probs[:, 1:].sum(dim=(2, 3)) + legacy_target[:, 1:].sum(dim=(2, 3))
    legacy_loss = 1 - ((2 * legacy_intersection + 1e-7) / (legacy_denominator + 1e-7)).mean()
    legacy_loss.backward()

    torch.testing.assert_close(sparse_loss, legacy_loss)
    torch.testing.assert_close(sparse_logits.grad, legacy_logits.grad)


def test_focal_loss_is_lower_for_confident_correct_predictions() -> None:
    """FocalLoss should down-weight easy, correct pixels."""
    target = torch.ones(1, 4, 4, dtype=torch.long)
    confident = torch.full((1, 3, 4, 4), -5.0)
    confident[:, 1] = 5.0
    uncertain = torch.full((1, 3, 4, 4), -0.5)
    uncertain[:, 1] = 0.5

    loss_fn = FocalLoss(include_background=True, alpha=1.0, gamma=2.0)
    confident_loss = loss_fn(torch.nn.functional.log_softmax(confident, dim=1), target)
    uncertain_loss = loss_fn(torch.nn.functional.log_softmax(uncertain, dim=1), target)

    assert confident_loss < uncertain_loss


def test_focal_loss_can_ignore_background_pixels() -> None:
    """Foreground-only focal loss should average over nonzero target pixels."""
    log_pred = torch.nn.functional.log_softmax(torch.zeros(1, 3, 2, 2), dim=1)
    target = torch.tensor([[[0, 1], [0, 2]]])

    loss = FocalLoss(include_background=False)(log_pred, target)

    assert torch.isfinite(loss)
    assert loss > 0


def test_segmentor_loss_combines_enabled_components() -> None:
    """SegmentorLoss should return total, focal, and dice components."""
    pred_masks = torch.randn(2, 4, 8, 8)
    gt_indices = torch.randint(0, 4, (2, 8, 8))

    losses = SegmentorLoss(focal_cof=0.5, dice_cof=2.0)(pred_masks, gt_indices)

    expected_total = 0.5 * losses["focal"] + 2.0 * losses["dice"]
    assert set(losses) == {"total_loss", "focal", "dice"}
    torch.testing.assert_close(losses["total_loss"], expected_total)


def test_segmentor_loss_interpolates_ground_truth_to_prediction_size() -> None:
    """Ground-truth masks should be resized when model output resolution differs."""
    pred_masks = torch.randn(1, 3, 8, 8)
    gt_masks = torch.randint(0, 3, (1, 4, 4))

    losses = SegmentorLoss(focal_cof=1.0, dice_cof=1.0)(pred_masks, gt_masks)

    assert torch.isfinite(losses["total_loss"])


def test_segmentor_loss_zero_coefficients_return_zero_components() -> None:
    """Disabled components should produce device-local zero tensors."""
    pred_masks = torch.randn(1, 3, 4, 4)
    gt_masks = torch.randint(0, 3, (1, 4, 4))

    losses = SegmentorLoss(focal_cof=0.0, dice_cof=0.0)(pred_masks, gt_masks)

    torch.testing.assert_close(losses["total_loss"], torch.tensor(0.0))
    torch.testing.assert_close(losses["focal"], torch.tensor(0.0))
    torch.testing.assert_close(losses["dice"], torch.tensor(0.0))


def test_deprecated_segmentor_loss_coefficients_are_ignored() -> None:
    """Deprecated coefficients should not add legacy keys to the loss dict."""
    pred_masks = torch.randn(1, 3, 4, 4)
    gt_masks = torch.randint(0, 3, (1, 4, 4))
    loss_fn = SegmentorLoss(
        focal_cof=1.0,
        dice_cof=1.0,
        objectness_penalty_cof=1.0,
        background_aware_cof=1.0,
    )

    losses = loss_fn(pred_masks, gt_masks)

    assert set(losses) == {"total_loss", "focal", "dice"}


def test_weighted_cross_entropy_consumes_indices_directly() -> None:
    """Median-frequency weights should match direct index-target cross-entropy."""
    logits = torch.randn(2, 3, 4, 5)
    targets = torch.randint(0, 3, (2, 4, 5))
    loss_fn = MedianFrequencyWeightedCrossEntropyLoss()
    loss_fn.set_class_pixel_statistics(
        torch.tensor([10, 20, 30]),
        torch.tensor([40, 40, 40]),
    )

    losses = loss_fn(logits, targets)
    expected = torch.nn.functional.cross_entropy(
        logits,
        targets,
        weight=loss_fn.class_weights.to(dtype=logits.dtype),
    )

    torch.testing.assert_close(losses["total_loss"], expected)
    torch.testing.assert_close(losses["cross_entropy"], expected)

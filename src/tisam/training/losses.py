"""TiSAM segmentation objectives configured by training.runner from tile statistics."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from prettyterm import get_logger
from torch.utils.data import DataLoader

from tisam.data.targets import (
    SUPERVISION_IGNORE_INDEX,
    resize_target_indices,
    target_supervision_validity,
)

logger = get_logger(__name__)


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return a graph-connected mean that is exactly zero for an empty mask."""
    weights = valid.to(device=values.device, dtype=values.dtype)
    weighted_sum = (values * weights).sum()
    count = weights.sum()
    return torch.where(count > 0, weighted_sum / count.clamp_min(1), weighted_sum)


class DiceLoss(nn.Module):
    def __init__(
        self,
        include_background=False,
    ):
        """Initialize the Dice loss used by `SegmentorLoss`.

        Args:
            include_background: If True, compute dice loss over all classes including background.
                If False, compute dice loss only on foreground classes (exclude index 0).
        """
        super().__init__()
        self.include_background = include_background

    def forward(
        self,
        pred_probs: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dice loss from pre-computed probabilities.

        `SegmentorLoss.forward` calls this after converting TiSAM logits to
        probabilities. Overlap is accumulated per class from the class-index
        target instead of a materialized one-hot tensor; supervision-ignored
        pixels contribute to neither support term.

        Args:
            pred_probs: (B, C, H, W) predicted probabilities (already softmaxed)
            target_indices: (B, H, W) ground truth class indices, where
                `SUPERVISION_IGNORE_INDEX` marks unsupervised pixels

        Returns:
            Scalar dice loss value
        """
        smooth = 1e-7

        batch_size, num_classes, _, _ = pred_probs.shape
        valid = target_supervision_validity(target_indices)
        flat_valid = valid.flatten(1)
        # Keep the full BCN tensor in its autocast dtype and accumulate in
        # float32; only the gathered BN terms are widened.
        flat_probs = pred_probs.flatten(2) * flat_valid.unsqueeze(1).to(pred_probs.dtype)
        flat_targets = target_indices.flatten(1).clamp_min(0)
        predicted_support = flat_probs.sum(dim=2, dtype=torch.float32)
        true_class_probs = (
            flat_probs.gather(dim=1, index=flat_targets.unsqueeze(1)).squeeze(1).float()
        )
        intersection = torch.zeros(
            (batch_size, num_classes),
            device=pred_probs.device,
            dtype=torch.float32,
        ).scatter_add(
            dim=1,
            index=flat_targets,
            src=true_class_probs,
        )
        target_support = torch.zeros_like(intersection).scatter_add(
            dim=1,
            index=flat_targets,
            src=flat_valid.to(dtype=torch.float32),
        )
        denominator = predicted_support + target_support

        if not self.include_background:
            intersection = intersection[:, 1:]
            denominator = denominator[:, 1:]

        dice = (2 * intersection + smooth) / (denominator + smooth)
        per_sample_loss = 1 - dice.mean(dim=1)
        valid_samples = valid.flatten(1).any(dim=1)
        return masked_mean(per_sample_loss, valid_samples)


class FocalLoss(nn.Module):
    def __init__(self, include_background=True, alpha=1, gamma=2):
        """Initialize the focal loss used by `SegmentorLoss`.

        Args:
            include_background: If True, compute loss over all classes including background.
                               If False, compute loss only on foreground classes (exclude index 0).
            alpha: Focal loss alpha parameter (weighting factor).
            gamma: Focal loss gamma parameter (focusing parameter).
        """
        super().__init__()
        self.include_background = include_background
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        log_pred: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss from pre-computed log probabilities.

        `SegmentorLoss.forward` calls this after computing TiSAM log-softmax
        predictions.

        Args:
            log_pred: (B, C, H, W) log probabilities (already log_softmaxed)
            target_indices: (B, H, W) ground truth class indices, where
                `SUPERVISION_IGNORE_INDEX` marks unsupervised pixels

        Returns:
            Scalar focal loss value
        """
        # Compute nll_loss (negative log likelihood) from log probabilities.
        # Unsupervised pixels contribute zero and are excluded from the mean.
        nll_loss = torch.nn.functional.nll_loss(
            log_pred,
            target_indices,
            reduction="none",
            ignore_index=SUPERVISION_IGNORE_INDEX,
        )

        # Compute p_t (probability of true class) from nll_loss
        # nll_loss = -log(p_t) => p_t = exp(-nll_loss)
        p_t = torch.exp(-nll_loss)

        # Compute focal loss: alpha * (1 - p_t)^gamma * nll_loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * nll_loss

        valid = target_supervision_validity(target_indices)
        # Mask out background pixels if include_background=False
        if not self.include_background:
            # Create mask for foreground pixels (where target is not class 0)
            valid = valid & (target_indices != 0)

        return masked_mean(focal_loss, valid)


class SegmentorLoss(nn.Module):
    """Combine focal and Dice losses for TiSAM segmentation training.

    `tisam.cli.train` instantiates this class and passes it to
    TiSAM's architecture step provider, which calls `forward` for each
    optimization step.
    """

    def __init__(
        self,
        focal_cof: float,
        dice_cof: float,
        objectness_penalty_cof: float | None = None,
        background_aware_cof: float | None = None,
    ):
        """Initialize the TiSAM segmentation loss.

        Args:
            focal_cof: Coefficient for focal loss
            dice_cof: Coefficient for dice loss
            objectness_penalty_cof: Deprecated and ignored.
            background_aware_cof: Deprecated and ignored.
        """
        super().__init__()
        self.focal_cof = focal_cof
        self.dice_cof = dice_cof

        if objectness_penalty_cof is not None:
            logger.warning(
                "`objectness_penalty_cof` is deprecated, ignored, and will be "
                "removed in a future release."
            )
        if background_aware_cof is not None:
            logger.warning(
                "`background_aware_cof` is deprecated, ignored, and will be "
                "removed in a future release."
            )

        # Initialize loss functions
        self.dice_loss_fn = DiceLoss(include_background=False)
        self.focal_loss_fn = FocalLoss(include_background=True)

        logger.info(f"Loss initialized: focal={focal_cof}, dice={dice_cof}")

    def forward(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-component loss for multiclass segmentation.

        Architecture steps call this with TiSAM predictions and class-index masks
        and log the returned loss dictionary.

        Args:
            pred_masks: Predicted logits tensor of shape (B, num_classes+1, H, W)
            gt_masks: Ground truth int64 class indices of shape (B, H, W)

        Returns:
            Dictionary of losses
        """
        target_indices = resize_target_indices(gt_masks, pred_masks.shape[-2:])

        # =========================================
        # Shared computation: compute once, use everywhere
        # =========================================
        # 1. log_softmax for focal loss (and numerical stability)
        log_pred = torch.nn.functional.log_softmax(pred_masks, dim=1)

        # 2. softmax probabilities for dice
        pred_probs = torch.exp(log_pred)

        # =========================================
        # Compute individual losses with shared tensors
        # =========================================
        dice_loss = (
            self.dice_loss_fn(pred_probs, target_indices)
            if self.dice_cof != 0
            else pred_masks.sum() * 0
        )
        focal_loss = (
            self.focal_loss_fn(log_pred, target_indices)
            if self.focal_cof != 0
            else pred_masks.sum() * 0
        )

        # Combine losses
        total_loss = self.focal_cof * focal_loss + self.dice_cof * dice_loss

        loss_dict = {
            "total_loss": total_loss,
            "focal": focal_loss,
            "dice": dice_loss,
        }

        return loss_dict


class MedianFrequencyWeightedCrossEntropyLoss(nn.Module):
    """Apply MATLAB-style median-frequency weighting to dense logits.

    Training component setup calls `configure_from_loader` before Mask2Former,
    DeepLab, HRNetV2, or U-Net steps consume the scalar loss dictionary.
    """

    def __init__(self) -> None:
        """Initialize the loss without loader-derived class weights."""
        super().__init__()
        self.register_buffer("class_weights", torch.empty(0), persistent=False)
        self.register_buffer("class_pixel_counts", torch.empty(0), persistent=False)
        self.register_buffer("class_image_pixel_counts", torch.empty(0), persistent=False)
        self.register_buffer(
            "supervision_ignored_class_ids",
            torch.empty(0, dtype=torch.int64),
            persistent=False,
        )

    @property
    def has_class_weights(self) -> bool:
        """Return whether training-label class weights have been configured."""
        return bool(self.class_weights.numel())

    def set_class_pixel_statistics(
        self,
        pixel_counts: torch.Tensor,
        image_pixel_counts: torch.Tensor,
        supervision_ignored_class_ids: tuple[int, ...] = (),
    ) -> None:
        """Compute MATLAB median-frequency weights from class pixel statistics."""
        counts = pixel_counts.detach().to(device="cpu")
        image_counts = image_pixel_counts.detach().to(device="cpu")
        if counts.ndim != 1:
            raise ValueError(f"Expected 1D class pixel counts, got shape {tuple(counts.shape)}.")
        if image_counts.ndim != 1:
            raise ValueError(
                f"Expected 1D class image-pixel counts, got shape {tuple(image_counts.shape)}."
            )
        if counts.shape != image_counts.shape:
            raise ValueError(
                "Class pixel counts and image-pixel counts must have the same shape, "
                f"got {tuple(counts.shape)} and {tuple(image_counts.shape)}."
            )
        if not torch.all(torch.isfinite(counts)):
            raise ValueError("Class pixel counts must be finite.")
        if not torch.all(torch.isfinite(image_counts)):
            raise ValueError("Class image-pixel counts must be finite.")
        if torch.any(counts < 0):
            raise ValueError("Class pixel counts must be non-negative.")
        if torch.any(image_counts < 0):
            raise ValueError("Class image-pixel counts must be non-negative.")
        total_pixels = counts.sum()
        if total_pixels <= 0:
            raise ValueError("Cannot configure weighted loss from an empty training dataset.")

        ignored = torch.zeros_like(counts, dtype=torch.bool)
        if supervision_ignored_class_ids:
            ignored_indices = torch.tensor(
                supervision_ignored_class_ids,
                dtype=torch.long,
            )
            if torch.any(ignored_indices < 0) or torch.any(ignored_indices >= counts.numel()):
                raise ValueError("Supervision-ignored class ids must index the class-count arrays.")
            ignored[ignored_indices] = True
        absent = (counts == 0) & ~ignored
        if torch.any(absent):
            absent_count = int(absent.sum().item())
            raise ValueError(
                "MATLAB median-frequency weighting is undefined when declared classes "
                f"are absent from training labels; found {absent_count} absent class(es)."
            )
        if torch.any((image_counts <= 0) & ~ignored):
            raise ValueError("All supervised classes must have positive image-pixel counts.")
        if torch.any(counts > image_counts):
            raise ValueError("Class pixel counts cannot exceed class image-pixel counts.")

        counts = counts.to(dtype=torch.float64)
        image_counts = image_counts.to(dtype=torch.float64)
        frequencies = counts[~ignored] / image_counts[~ignored]
        median_frequency = torch.quantile(frequencies, 0.5)
        class_weights = torch.zeros_like(counts)
        class_weights[~ignored] = median_frequency / frequencies
        self.class_weights = class_weights
        self.class_pixel_counts = pixel_counts.detach().to(device="cpu", dtype=torch.int64)
        self.class_image_pixel_counts = image_pixel_counts.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        self.supervision_ignored_class_ids = torch.tensor(
            supervision_ignored_class_ids,
            dtype=torch.int64,
        )

        logger.info("Median-frequency class weights configured from training labels.")

    def manifest_provenance(self) -> dict[str, object]:
        """Return configured training-only class statistics for run manifests."""
        if not self.has_class_weights:
            raise RuntimeError(
                "Median-frequency weighted loss provenance requires configured class weights."
            )
        return {
            "name": "median_frequency_weighted_cross_entropy",
            "statistics_source": "configured_training_partition",
            "pixel_counts": self.class_pixel_counts.tolist(),
            "image_pixel_counts": self.class_image_pixel_counts.tolist(),
            "supervision_ignored_class_ids": self.supervision_ignored_class_ids.tolist(),
            "class_weights": self.class_weights.tolist(),
        }

    def configure_from_loader(self, train_loader: DataLoader[Any]) -> None:
        """Configure class weights from the shared TiSAM training DataLoader."""
        class_pixel_statistics = getattr(train_loader.dataset, "class_pixel_statistics", None)
        if not callable(class_pixel_statistics):
            raise TypeError(
                "Median-frequency weighted loss requires a training dataset with "
                "class_pixel_statistics()."
            )
        pixel_counts, image_pixel_counts = class_pixel_statistics()
        ignored_class_ids = tuple(
            getattr(
                train_loader.dataset,
                "supervision_ignored_global_class_ids",
                (),
            )
        )
        self.set_class_pixel_statistics(
            pixel_counts,
            image_pixel_counts,
            ignored_class_ids,
        )

    def forward(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return class-weighted dense cross-entropy for segmentation logits."""
        target_indices = resize_target_indices(gt_masks, pred_masks.shape[-2:])
        valid = target_supervision_validity(target_indices)
        class_weights = (
            self.class_weights.to(device=pred_masks.device, dtype=pred_masks.dtype)
            if self.has_class_weights
            else None
        )
        per_pixel_loss = torch.nn.functional.cross_entropy(
            pred_masks,
            target_indices,
            weight=class_weights,
            reduction="none",
            ignore_index=SUPERVISION_IGNORE_INDEX,
        )
        if class_weights is None:
            denominator = valid.sum().to(dtype=per_pixel_loss.dtype)
        else:
            denominator = (
                class_weights[target_indices.clamp_min(0)] * valid.to(dtype=class_weights.dtype)
            ).sum()
        weighted_sum = (per_pixel_loss * valid.to(dtype=per_pixel_loss.dtype)).sum()
        cross_entropy = torch.where(
            denominator > 0,
            weighted_sum / denominator.clamp_min(1),
            weighted_sum,
        )
        return {
            "total_loss": cross_entropy,
            "cross_entropy": cross_entropy,
        }


class TemperedWeightedCrossEntropyDiceBackgroundLoss(MedianFrequencyWeightedCrossEntropyLoss):
    """Balance semantic sensitivity with explicit tissue/background separation.

    `training.runner.build_training_loss` constructs this class for the
    ``tempered_weighted_ce_dice_bg`` mode. The inherited loader configuration
    receives real-tile class counts before training and tempers the resulting
    median-frequency weights.
    """

    def __init__(
        self,
        *,
        background_class_index: int,
        class_weight_power: float,
        weighted_ce_cof: float,
        dice_cof: float,
        tissue_background_cof: float,
    ) -> None:
        """Initialize the hybrid loss and its configured component weights."""
        super().__init__()
        if background_class_index <= 0:
            raise ValueError("background_class_index must identify a non-void class.")
        if not 0 < class_weight_power <= 1:
            raise ValueError("class_weight_power must be in (0, 1].")
        if weighted_ce_cof <= 0:
            raise ValueError("weighted_ce_cof must be positive.")
        if dice_cof < 0:
            raise ValueError("dice_cof must be non-negative.")
        if tissue_background_cof < 0:
            raise ValueError("tissue_background_cof must be non-negative.")

        self.background_class_index = background_class_index
        self.class_weight_power = class_weight_power
        self.weighted_ce_cof = weighted_ce_cof
        self.dice_cof = dice_cof
        self.tissue_background_cof = tissue_background_cof
        self.dice_loss_fn = DiceLoss(include_background=False)

    def set_class_pixel_statistics(
        self,
        pixel_counts: torch.Tensor,
        image_pixel_counts: torch.Tensor,
        supervision_ignored_class_ids: tuple[int, ...] = (),
    ) -> None:
        """Configure and temper median-frequency weights from real-tile counts."""
        super().set_class_pixel_statistics(
            pixel_counts,
            image_pixel_counts,
            supervision_ignored_class_ids,
        )
        self.class_weights = self.class_weights.pow(self.class_weight_power)
        logger.info(
            f"Tempered median-frequency class weights with power {self.class_weight_power:g}."
        )

    def _tissue_background_loss(
        self,
        pred_masks: torch.Tensor,
        target_indices: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Compute binary tissue-versus-nontissue CE over non-void pixels."""
        if self.background_class_index >= pred_masks.shape[1]:
            raise ValueError(
                "background_class_index must be smaller than the prediction "
                f"channel count, got {self.background_class_index} and "
                f"{pred_masks.shape[1]}."
            )
        tissue_logits = torch.cat(
            (
                pred_masks[:, 1 : self.background_class_index],
                pred_masks[:, self.background_class_index + 1 :],
            ),
            dim=1,
        )
        if tissue_logits.shape[1] == 0:
            raise ValueError("At least one semantic tissue class is required.")

        grouped_logits = torch.stack(
            (
                torch.logsumexp(tissue_logits, dim=1),
                pred_masks[:, self.background_class_index],
            ),
            dim=1,
        )
        grouped_targets = (target_indices == self.background_class_index).long()
        valid_nonvoid = target_indices != 0
        pixel_losses = torch.nn.functional.cross_entropy(
            grouped_logits,
            grouped_targets,
            reduction="none",
        )
        return masked_mean(pixel_losses, valid & valid_nonvoid)

    def forward(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return tempered weighted CE, semantic Dice, and background separation."""
        target_indices = resize_target_indices(gt_masks, pred_masks.shape[-2:])
        valid = target_supervision_validity(target_indices)
        weighted_ce = super().forward(pred_masks, gt_masks)["cross_entropy"]
        zero = pred_masks.sum() * 0.0
        dice_loss = (
            self.dice_loss_fn(torch.softmax(pred_masks, dim=1), target_indices)
            if self.dice_cof != 0
            else zero
        )
        tissue_background_loss = (
            self._tissue_background_loss(pred_masks, target_indices, valid)
            if self.tissue_background_cof != 0
            else zero
        )
        total_loss = (
            self.weighted_ce_cof * weighted_ce
            + self.dice_cof * dice_loss
            + self.tissue_background_cof * tissue_background_loss
        )
        return {
            "total_loss": total_loss,
            "tempered_weighted_ce": weighted_ce,
            "dice": dice_loss,
            "tissue_background": tissue_background_loss,
        }

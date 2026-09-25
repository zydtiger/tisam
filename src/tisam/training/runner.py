"""Single-process TiSAM training composed with Mammoth's Trainer lifecycle."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from mammoth.core import claim_logical_run_lease
from mammoth.core.artifacts import open_artifact_session
from mammoth.logging import RunObserver
from mammoth.torch import CheckpointSavePolicy, EarlyStopping, Trainer, WarmupLinearLR
from mammoth.torch import TrainerConfig as MammothConfig

from tisam.checkpointing.weights import apply_weights, config_from_metadata, read_checkpoint
from tisam.config.run import TrainConfig
from tisam.data.tile_dataset import get_tile_loaders
from tisam.model.segmentor import TiSAM

from .checkpoints import TrainingCheckpointPolicy
from .losses import (
    MedianFrequencyWeightedCrossEntropyLoss,
    SegmentorLoss,
    TemperedWeightedCrossEntropyDiceBackgroundLoss,
)
from .metrics import SegmentationIOUMetric
from .observability import artifact_fields, metric_routes, training_observer
from .steps import DenseSegmentationSteps


def build_loss(config: TrainConfig, loader):
    """Configure the selected existing TiSAM objective from raw training support."""
    loss: torch.nn.Module
    if config.loss == "focal_dice":
        loss = SegmentorLoss(config.focal_coefficient, config.dice_coefficient)
    elif config.loss == "tempered_weighted_ce_dice_bg":
        if "background" not in config.data.class_names:
            raise ValueError("The tempered objective requires an explicit background tissue class")
        loss = TemperedWeightedCrossEntropyDiceBackgroundLoss(
            background_class_index=config.data.class_names.index("background"),
            class_weight_power=config.tempered_weight_power,
            weighted_ce_cof=config.weighted_ce_coefficient,
            dice_cof=config.dice_coefficient,
            tissue_background_cof=config.tissue_background_coefficient,
        )
    else:
        loss = MedianFrequencyWeightedCrossEntropyLoss()
    if isinstance(loss, MedianFrequencyWeightedCrossEntropyLoss):
        loss.configure_from_loader(loader)
    return loss.to(config.device)


def latest_checkpoint(directory: Path) -> Path | None:
    """Select the highest completed epoch, never a metric-best weights file."""
    candidates = list(directory.glob("latest_epoch_*.pt")) + list(directory.glob("epoch_*.pt"))
    return max(candidates, key=lambda p: int(p.stem.rsplit("_", 1)[1])) if candidates else None


def train(
    config: TrainConfig,
    *,
    resume: bool = True,
    checkpoint: Path | None = None,
    initialize_from: Path | None = None,
    trusted: bool = False,
):
    """Train or resume one run; weight initialization starts a distinct run."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("TiSAM supports single-process training only")
    if not config.data.train or not config.data.validation:
        raise ValueError("Training requires nonempty train and validation sources")
    directory = config.out_dir / config.name
    directory.mkdir(parents=True, exist_ok=True)
    with claim_logical_run_lease(directory) as lease:
        return _train_locked(
            config,
            lease=lease,
            resume=resume,
            checkpoint=checkpoint,
            initialize_from=initialize_from,
            trusted=trusted,
        )


def _train_locked(config, *, lease, resume, checkpoint, initialize_from, trusted):
    """Select resume state and establish one attempt while holding run ownership."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    directory = config.out_dir / config.name
    checkpoint_dir = directory / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    selected = checkpoint or (latest_checkpoint(checkpoint_dir) if resume else None)
    if initialize_from is not None and selected is not None:
        raise ValueError("Weight initialization and training resume cannot be combined")
    if not resume and latest_checkpoint(checkpoint_dir) is not None:
        raise ValueError(
            "Fresh training requires a new run name; existing checkpoints are preserved"
        )
    initial_epoch = 0
    initial_step = 0
    resume_receipt = None
    if selected is not None:
        selected = selected.resolve()
        with open_artifact_session(selected) as artifact:
            with artifact.open_reader() as reader:
                payload = torch.load(reader, map_location="cpu", weights_only=True)
            resume_receipt = artifact.receipt
        if payload.get("training_schema") != 1:
            raise ValueError("Use initialize_from for historical model weights")
        initial_epoch = payload["epoch"] + 1
        initial_step = payload["global_step"]
        del payload
    with training_observer(
        config,
        lease=lease,
        resume_receipt=resume_receipt,
        initial_epoch=initial_epoch,
        initial_step=initial_step,
    ) as observer:
        return _fit(
            config,
            observer,
            checkpoint_dir=checkpoint_dir,
            selected=selected,
            resume_receipt=resume_receipt,
            initial_epoch=initial_epoch,
            initialize_from=initialize_from,
            trusted=trusted,
        )


def _fit(
    config: TrainConfig,
    observer: RunObserver,
    *,
    checkpoint_dir,
    selected,
    resume_receipt,
    initial_epoch,
    initialize_from,
    trusted,
):
    """Build project components inside the attempt's logging lifetime."""
    train_loader, val_loader = get_tile_loaders(config, initial_epoch=initial_epoch)
    if not len(train_loader) or not len(val_loader):
        raise ValueError("Train and validation loaders must contain batches")
    model = TiSAM(config.model, sam3_checkpoint=config.sam3_checkpoint).to(config.device)
    if initialize_from is not None:
        with open_artifact_session(initialize_from.resolve()) as artifact:
            state, metadata = read_checkpoint(initialize_from, trusted=trusted)
            initialization_receipt = artifact.receipt
        saved = config_from_metadata(metadata)
        if saved is not None and saved != config.model:
            raise ValueError("Initialization checkpoint disagrees with model configuration")
        apply_weights(model, state)
        observer.emit(
            "task_completed",
            phase="train",
            task_id="initialization-source",
            artifacts=[artifact_fields(initialization_receipt)],
        )
    loss = build_loss(config, train_loader)
    optimizer = torch.optim.AdamW(
        model.get_optimizer_param_groups(
            config.learning_rate, config.image_lr_ratio, config.extra_lr_ratio
        ),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = math.ceil(len(train_loader) / config.accumulation_steps) * config.epochs
    scheduler = WarmupLinearLR(optimizer, warmup_ratio=config.warmup_ratio, total_steps=total_steps)
    early_stopping = EarlyStopping(
        metric=config.early_stopping_monitor,
        mode=config.early_stopping_mode,
        patience=config.patience,
        min_delta=config.min_delta,
    )
    policy = TrainingCheckpointPolicy(model, optimizer, scheduler, early_stopping, config)
    policy.resume_receipt = resume_receipt
    steps = DenseSegmentationSteps(loss)
    evaluated = sorted(
        {
            config.data.class_names.index(name)
            for s in config.data.validation
            for raw, name in enumerate(s.metadata.label_to_class_name)
            if raw not in s.metadata.metric_ignored_raw_labels
        }
    )
    if config.compile:
        model.setup_compile()
    with Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
        train_step=steps.train_step,
        validation_step=steps.validation_step,
        config=MammothConfig(
            epochs=config.epochs,
            device=config.device,
            precision=config.precision,
            strategy="single",
            gradient_accumulation_steps=config.accumulation_steps,
            max_gradient_norm=1.0,
            scheduler_interval="optimizer",
            emit_fit_phase_events=False,
        ),
        scheduler=scheduler,
        observer=observer,
        train_metric_routes=metric_routes(config, validation=False),
        validation_metric_routes=metric_routes(config, validation=True),
        callbacks=(early_stopping,),
        validation_stateful_metrics={
            "segmentation_iou": SegmentationIOUMetric(
                num_classes=len(config.data.class_names),
                device=config.device,
                class_names=config.data.class_names,
                evaluated_class_ids=evaluated,
            )
        },
        checkpoint_dir=checkpoint_dir,
        checkpoint_policy=policy,
        checkpoint_save_policy=CheckpointSavePolicy(
            mode=config.checkpoint_mode, save_best=True, every_epochs=1
        ),
    ) as trainer:
        policy.scaler = trainer.scaler
        if selected is not None:
            trainer.load_checkpoint(selected)
        return trainer.fit()

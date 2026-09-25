"""Compose Mammoth execution records and artifact receipts for one training attempt."""

from __future__ import annotations

import logging
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from typing import Iterator

from mammoth.core import LogicalRunLease
from mammoth.core.artifacts import ArtifactReceipt, atomic_write_json, inspect_artifact
from mammoth.core.events import ExecutionEventWriter
from mammoth.core.execution import (
    create_execution_context,
    generate_execution_id,
    latest_execution_id,
)
from mammoth.execution import ExecutionSession
from mammoth.logging import (
    ExecutionLogging,
    JsonlEventSink,
    RunObserver,
    create_process_text_handler,
)
from mammoth.logging.model import Observation
from mammoth.logging.tensorboard import TensorBoardSink
from mammoth.torch import MetricRoute

from tisam.config.run import TrainConfig


def metric_routes(config: TrainConfig, *, validation: bool) -> dict[str, MetricRoute]:
    """Route every supported loss and IoU summary to a phase-specific epoch series."""
    losses = {
        "median_frequency_weighted_ce": ("cross_entropy",),
        "focal_dice": ("focal", "dice"),
        "tempered_weighted_ce_dice_bg": ("tempered_weighted_ce", "dice", "tissue_background"),
    }
    names = list(losses[config.loss])
    if validation:
        names += ["total_loss", "bg_iou", "mean_fg_iou"]
        names += [f"iou_{name}" for name in config.data.class_names]
        return {name: MetricRoute(None, f"validation/{name}") for name in names}
    names += ["loss"]
    return {name: MetricRoute(name, f"train/{name}") for name in names}


class TrainingJsonlSink(JsonlEventSink):
    """Retain epoch metrics and explicit batch rates alongside Mammoth's native rates."""

    def observe(self, observation: Observation) -> None:
        fields = dict(observation.fields)
        # KeyboardInterrupt and some setup exceptions have no message; v1 omits empties.
        if fields.get("message") == "":
            fields.pop("message")
        if observation.event == "task_completed" and observation.metrics:
            fields["epoch_metrics"] = dict(observation.metrics)
        if observation.event == "progress" and fields.get("throughput") is not None:
            rate = fields["throughput"]
            if fields.get("phase") == "train":
                # Mammoth counts accumulation windows; the last may be shorter.
                completed = fields.get("completed", 0)
                batch = fields.get("coordinates", {}).get("batch")
                if completed and batch is not None:
                    fields["batches_per_second"] = rate * (batch + 1) / completed
                fields["throughput_unit"] = "accumulation_windows_per_second"
            elif fields.get("phase") == "validation":
                fields["batches_per_second"] = rate
                fields["throughput_unit"] = "batches_per_second"
        super().observe(replace(observation, fields=fields))


def artifact_fields(receipt: ArtifactReceipt) -> dict:
    """Serialize exact-byte provenance without loading the artifact again."""
    return {
        "path": str(receipt.path),
        "size_bytes": receipt.size_bytes,
        "sha256": receipt.sha256,
    }


@contextmanager
def training_observer(
    config: TrainConfig,
    *,
    lease: LogicalRunLease,
    resume_receipt: ArtifactReceipt | None,
    initial_epoch: int,
    initial_step: int,
) -> Iterator[RunObserver]:
    """Compose TiSAM provenance and metrics inside a Mammoth execution session."""
    directory = (config.out_dir / config.name).resolve()
    execution_id = generate_execution_id()
    snapshot = directory / "logs" / "executions" / execution_id / "config.json"
    context = create_execution_context(
        directory,
        run_name=config.name,
        invocation_kind="train",
        intended_phases=("train", "validation"),
        world_size=1,
        execution_mode="single",
        command=sys.argv,
        execution_id=execution_id,
        config_reference=snapshot,
        previous_execution_id=latest_execution_id(directory),
        resume_checkpoint=None if resume_receipt is None else resume_receipt.path,
        resume_checkpoint_sha256=None if resume_receipt is None else resume_receipt.sha256,
        starting_epoch=initial_epoch,
        starting_global_step=initial_step,
        runtime={"device": config.device, "precision": config.precision},
    )
    with ExitStack() as resources:
        handler = create_process_text_handler(context, rank=0)
        resources.callback(handler.close)
        writer = ExecutionEventWriter.for_process(context, rank=0)
        resources.callback(writer.close)
        if not writer.enabled:
            raise RuntimeError("Mammoth JSONL logging could not open its event stream")
        tensorboard = TensorBoardSink(context.execution_dir / "tensorboard")
        resources.callback(tensorboard.close)
        observer = RunObserver((TrainingJsonlSink(writer), tensorboard))
        resources.callback(observer.close)
        execution_logging = ExecutionLogging(context, 0, observer, handler, writer)
        root = logging.getLogger()
        previous_level = root.level
        root.addHandler(handler)
        resources.callback(root.removeHandler, handler)
        root.setLevel(min(previous_level, logging.INFO))
        resources.callback(root.setLevel, previous_level)
        with (
            ExecutionSession.from_established(
                context, execution_logging, logical_run_lease=lease
            ) as session,
            session.phase_scope("train"),
        ):
            try:
                atomic_write_json(snapshot, config.model_dump(mode="json"))
                receipt = inspect_artifact(snapshot)
                observer.emit(
                    "task_completed",
                    phase="train",
                    task_id="configuration",
                    artifacts=[artifact_fields(receipt)],
                )
                # Preserve the historical convenience path; attempt snapshots are immutable.
                atomic_write_json(directory / "config.json", config.model_dump(mode="json"))
                if resume_receipt is not None:
                    observer.emit(
                        "task_completed",
                        phase="train",
                        task_id="resume-source",
                        artifacts=[artifact_fields(resume_receipt)],
                    )
                with observer.periodic_heartbeats(phase="train"):
                    yield observer
            except KeyboardInterrupt:
                logging.getLogger(__name__).warning("Training interrupted")
                raise
            except BaseException:
                logging.getLogger(__name__).exception("Training attempt failed")
                raise

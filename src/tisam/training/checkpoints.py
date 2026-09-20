"""TiSAM training-state policy for Mammoth atomic checkpoint publication."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from mammoth.torch import (
    CheckpointInspection,
    RestoreOptions,
    TrainerCheckpointRestore,
    TrainerCheckpointWriters,
)
from safetensors.torch import save_file

from tisam.checkpointing.weights import apply_weights, checkpoint_state, model_metadata


def rng_state() -> dict:
    """Capture random generators in weights-only-loadable form."""
    np_state = cast(tuple[Any, ...], np.random.get_state(legacy=True))
    return {
        "python": random.getstate(),
        "numpy": [np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), *np_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    """Restore random streams after model and loader construction."""
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


class TrainingCheckpointPolicy:
    """Keep TiSAM meaning local while Mammoth owns save and restore mechanics."""

    def __init__(self, model, optimizer, scheduler, early_stopping, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.config = config
        self.scaler = None
        self.metadata = model_metadata(
            config.model,
            class_names=config.data.class_names,
            mean=config.data.mean,
            std=config.data.std,
        )

    def read(self, path: Path) -> dict:
        """Reject historical training state and changed resume contracts."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("training_schema") != 1:
            raise ValueError(
                "Cross-project training resume is unsupported; import model weights instead"
            )
        saved = dict(payload["training_config"])
        active = self.config.model_dump(mode="json")
        # Epoch horizon is preserved because it defines scheduler semantics.
        for field in (
            "out_dir",
            "name",
            "device",
            "num_workers",
            "persistent_workers",
            "compile",
            "sam3_checkpoint",
        ):
            saved.pop(field, None)
            active.pop(field, None)
        if saved != active:
            raise ValueError(
                "Training configuration changed; use weight initialization for a new run"
            )
        return payload

    def inspect(self, path: Path) -> CheckpointInspection:
        """Check resumability before mutating training state."""
        self.read(path)
        return CheckpointInspection(
            frozenset(
                {"model", "optimizer", "scheduler", "callbacks", "trainer", "scaler", "project"}
            )
        )

    def restore(
        self, path: Path, *, device: torch.device, options: RestoreOptions
    ) -> TrainerCheckpointRestore:
        """Load model/RNG/scaler state and return generic components to Mammoth."""
        del device, options
        payload = self.read(path)
        apply_weights(self.model, payload["model_state_dict"])
        restore_rng(payload["rng_state"])
        if self.scaler is not None:
            self.scaler.load_state_dict(payload["scaler"])
        return TrainerCheckpointRestore(
            epoch=payload["epoch"],
            optimizer_step=payload["optimizer_step"],
            global_step=payload["global_step"],
            stopped_early=payload["stopped_early"],
            optimizer_state_dict=payload["optimizer"],
            scheduler_state_dict=payload["scheduler"],
            callback_state_dicts={0: payload["early_stopping"]},
            restored_components=frozenset({"model", "scaler", "project"}),
        )

    def capture(self, context) -> TrainerCheckpointWriters:
        """Snapshot independently of the asynchronous checkpoint serializer."""
        state = checkpoint_state(self.model)
        payload = {
            "training_schema": 1,
            "model_state_dict": state,
            "checkpoint_metadata": self.metadata,
            "training_config": self.config.model_dump(mode="json"),
            "epoch": context.epoch,
            "optimizer_step": context.optimizer_step,
            "global_step": context.global_step,
            "stopped_early": context.stopped_early,
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler": copy.deepcopy(self.scheduler.state_dict()),
            "early_stopping": copy.deepcopy(self.early_stopping.state_dict()),
            "rng_state": rng_state(),
            "scaler": copy.deepcopy(self.scaler.state_dict()) if self.scaler is not None else {},
        }
        tensors = {k: v for k, v in state.items() if not v.is_complex()}
        return TrainerCheckpointWriters(
            resumable=lambda path: torch.save(payload, path),
            best=lambda path: save_file(tensors, str(path), metadata=self.metadata),
        )

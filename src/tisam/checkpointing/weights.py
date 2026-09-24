"""Model-only checkpoint import and export, independent of training runtimes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tisam.config.model import ModelConfig
from tisam.config.weights import SAM3_HF_INITIALIZATION_IDENTITY
from tisam.model.segmentor import TiSAM

MODEL_CONFIG_KEY = "tisam_model_config"
SCHEMA_KEY = "tisam_schema"


def legacy_model_config(payload: Mapping[str, Any]) -> ModelConfig:
    """Translate architecture fields from historical dual-encoder checkpoints."""
    if payload.get("model_architecture", "mask2former") != "mask2former":
        raise ValueError("Only dual-encoder TiSAM checkpoints are supported")
    if payload.get("model_mask2former_encoder", "sam3") != "sam3" or not payload.get(
        "model_use_sam3_encoder", True
    ):
        raise ValueError("Checkpoint is not a dual-encoder TiSAM model")
    required = (
        "dataset_num_classes",
        "model_extra_encoder",
        "model_total_queries",
        "model_num_layers",
        "model_d_model",
    )
    if any(k not in payload for k in required):
        raise ValueError("Checkpoint configuration is incomplete; provide an explicit ModelConfig")
    aliases = {
        "num_classes": "dataset_num_classes",
        "input_hw": "dataset_image_hw",
        "output_hw": "dataset_image_hw",
        "finetune": "model_image_finetune",
        "finetune_last_n_blocks": "model_image_finetune_last_n_blocks",
        "finetune_neck_convs": "model_image_finetune_neck_convs",
    }
    values = {}
    for field in ModelConfig.model_fields:
        old_name = aliases.get(field, "model_" + field)
        if old_name in payload:
            values[field] = payload[old_name]
    if "pos_embed_mode" not in values:
        values["pos_embed_mode"] = "rope" if payload.get("model_use_pos_embed", True) else "none"
    return ModelConfig.model_validate(values)


def read_checkpoint(
    path: str | Path, *, trusted: bool = False
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read weights and metadata; legacy pickled files require explicit trust."""
    path = Path(path)
    if path.suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as handle:
            return {k: handle.get_tensor(k) for k in handle.keys()}, dict(handle.metadata() or {})
    if path.suffix != ".pt":
        raise ValueError("Checkpoint must be .pt or .safetensors")
    payload = torch.load(path, map_location="cpu", weights_only=not trusted)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a mapping")
    state = payload.get("model_state_dict", payload.get("model", payload))
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Checkpoint contains no tensor state dictionary")
    return state, dict(payload.get("checkpoint_metadata", {}))


def config_from_metadata(metadata: Mapping[str, Any]) -> ModelConfig | None:
    """Resolve native or historical configuration without a research checkout."""
    if MODEL_CONFIG_KEY in metadata:
        return ModelConfig.model_validate_json(metadata[MODEL_CONFIG_KEY])
    for key in ("tisam_config_payload", "tisam_checkpoint_config_payload"):
        if key in metadata:
            payload = metadata[key]
            return legacy_model_config(json.loads(payload) if isinstance(payload, str) else payload)
    return None


def apply_weights(model: TiSAM, state: Mapping[str, torch.Tensor]) -> None:
    """Allow only omitted frozen encoder parameters and reconstructible buffers."""
    expected = model.state_dict()
    frozen = {
        n
        for n, p in model.named_parameters(remove_duplicate=False)
        if not p.requires_grad and n.startswith(("image_encoder.", "extra_encoder."))
    }
    # Historical RoPE complex buffers are deterministically recreated at construction.
    reconstructible = {
        n for n, value in model.named_buffers(remove_duplicate=False) if value.is_complex()
    }
    legacy_projection_names = {
        f"mask_decoder.transformer_decoder.pos_enc_proj.{level}.{parameter}"
        for level in range(4)
        for parameter in ("weight", "bias")
    }
    ignored_legacy = (set(state) - set(expected)) & legacy_projection_names
    incoming = {k: v for k, v in state.items() if k not in ignored_legacy}
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        available = [name for name in names if name in incoming]
        if not available:
            continue
        value = incoming[available[0]]
        if any(not torch.equal(value, incoming[name]) for name in available[1:]):
            raise ValueError("Conflicting checkpoint values for shared parameter aliases")
        for name in names:
            incoming.setdefault(name, value)
    missing = set(expected) - set(incoming) - frozen - reconstructible
    unexpected = set(incoming) - set(expected)
    mismatched = [
        k for k in set(expected) & set(incoming) if expected[k].shape != incoming[k].shape
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Incompatible TiSAM weights: missing={sorted(missing)}, unexpected={sorted(unexpected)}, shape={sorted(mismatched)}"
        )
    model.load_state_dict(incoming, strict=False)


def load_model(
    path: str | Path,
    *,
    config: ModelConfig | None = None,
    device: str = "cpu",
    sam3_checkpoint: Path | None = None,
    trusted: bool = False,
) -> TiSAM:
    """Restore weights after rebuilding the pretrained encoders; return eval mode."""
    state, metadata = read_checkpoint(path, trusted=trusted)
    saved = config_from_metadata(metadata)
    if config is None:
        config = saved
    elif saved is not None and config != saved:
        raise ValueError("Explicit ModelConfig disagrees with checkpoint architecture")
    if config is None:
        raise ValueError("Checkpoint has no model configuration; pass config=ModelConfig(...)")
    model = TiSAM(config, sam3_checkpoint=sam3_checkpoint)
    apply_weights(model, state)
    model.to(device).eval()
    return model


def model_metadata(
    config: ModelConfig,
    *,
    class_names: list[str] | None = None,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> dict[str, str]:
    """Record model identity and preprocessing for portable weights."""
    return {
        SCHEMA_KEY: "1",
        "sam3_identity": SAM3_HF_INITIALIZATION_IDENTITY,
        MODEL_CONFIG_KEY: config.model_dump_json(),
        "class_names": json.dumps(class_names),
        "mean": json.dumps(mean),
        "std": json.dumps(std),
    }


def checkpoint_state(model: TiSAM) -> dict[str, torch.Tensor]:
    """Capture trainable parameters and persistent buffers, omitting frozen weights."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad} | {
        n for n, _ in model.named_buffers()
    }
    return {
        n: v.detach().cpu().contiguous().clone() for n, v in model.state_dict().items() if n in keep
    }


def export_weights(
    model: TiSAM, path: str | Path, *, metadata: Mapping[str, str] | None = None
) -> None:
    """Write TiSAM weights for model-only consumers; encoders remain external."""
    state = checkpoint_state(model)
    state = {n: v for n, v in state.items() if not v.is_complex()}
    save_file(state, str(path), metadata=dict(metadata or model_metadata(model.config)))

"""Single-process, resumable WSI inference with bounded-memory TIFF output."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import tifffile
import torch
import zarr
from torch.utils.data import DataLoader, Subset

from tisam.data.wsi_dataset import WSIDataset, read_segmentation_source_shape
from tisam.model.segmentor import TiSAM


def model_fingerprint(model: TiSAM) -> str:
    """Identify exact weights before reusing previously computed WSI patches."""
    digest = hashlib.sha256(model.config.model_dump_json().encode())
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def segment_wsi(
    model: TiSAM,
    source: str | Path,
    output: str | Path,
    *,
    patch_size: int = 1024,
    effective_size: int = 800,
    batch_size: int = 1,
    num_workers: int = 0,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    resume: bool = True,
) -> Path:
    """Segment TIFF/PNG into a tiled TIFF, retaining scratch on interruption.

    A sibling .work directory contains a chunked class mask and completion bitmap.
    Resume checks input identity, preprocessing, geometry and model weights.
    Final output is published atomically; successful scratch is reproducible and
    removed. Existing outputs are never overwritten.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("output must be a distinct TIFF path")
    if effective_size <= 0 or patch_size < effective_size or (patch_size - effective_size) % 2:
        raise ValueError("patch_size-effective_size must be nonnegative and even")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if any(v <= 0 for v in std):
        raise ValueError("std must be positive")
    if output.exists():
        raise FileExistsError(output)
    height, width, _ = read_segmentation_source_shape(source)
    stat = source.stat()
    identity = {
        "schema": 1,
        "source": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "shape": [height, width],
        "model": model_fingerprint(model),
        "patch_size": patch_size,
        "effective_size": effective_size,
        "mean": list(mean),
        "std": list(std),
    }
    work = output.with_name(output.name + ".work")
    output.parent.mkdir(parents=True, exist_ok=True)
    if work.is_symlink():
        raise ValueError("WSI scratch directory cannot be a symlink")
    if work.exists():
        if not resume:
            raise FileExistsError(f"Scratch exists; resume it or choose another output: {work}")
        saved = json.loads((work / "identity.json").read_text())
        if saved != identity:
            raise ValueError("WSI resume identity differs; choose another output")
    else:
        work.mkdir()
        (work / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    # An exclusive lock prevents concurrent writers from corrupting recovery state.
    lock = work / "writer.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    was_training = model.training
    try:
        with WSIDataset(source, mean, std, patch_size, effective_size) as dataset:
            group = zarr.open_group(work / "patches.zarr", mode="a", zarr_format=2)
            dtype = "uint8" if model.total_classes <= 256 else "uint16"
            mask = group.require_array(
                "mask",
                shape=(height, width),
                chunks=(effective_size, effective_size),
                dtype=dtype,
                fill_value=0,
            )
            completed = group.require_array(
                "completed", shape=(len(dataset),), chunks=(1,), dtype="bool", fill_value=False
            )
            pending = np.flatnonzero(~np.asarray(completed[:], dtype=bool)).tolist()
            loader = DataLoader(
                Subset(dataset, pending),
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=False,
            )
            device = next(model.parameters()).device
            padding = (patch_size - effective_size) // 2
            model.eval()
            offset = 0
            with torch.inference_mode():
                for images in loader:
                    logits = model(images.to(device))
                    logits = torch.nn.functional.interpolate(
                        logits, size=(patch_size, patch_size), mode="bilinear", align_corners=False
                    )
                    cores = (
                        logits[
                            :,
                            :,
                            padding : padding + effective_size,
                            padding : padding + effective_size,
                        ]
                        .argmax(1)
                        .cpu()
                        .numpy()
                        .astype(dtype)
                    )
                    for core in cores:
                        index = pending[offset]
                        y, x = map(int, dataset.positions[index])
                        h, w = min(effective_size, height - y), min(effective_size, width - x)
                        mask[y : y + h, x : x + w] = core[:h, :w]
                        completed[index] = True
                        offset += 1
            if not bool(np.all(completed[:])):
                raise RuntimeError("WSI publication requires every patch")
            temporary = work / "mask.tif"

            def tiles():
                for y in range(0, height, 256):
                    for x in range(0, width, 256):
                        yield np.asarray(mask[y : min(y + 256, height), x : min(x + 256, width)])

            tifffile.imwrite(
                temporary,
                data=tiles(),
                shape=(height, width),
                dtype=dtype,
                tile=(256, 256),
                compression="zstd",
                photometric="minisblack",
                bigtiff=True,
            )
            # Hard-link publication refuses a destination created by another writer.
            os.link(temporary, output)
    finally:
        model.train(was_training)
        os.close(descriptor)
    shutil.rmtree(work)
    return output

"""Tile datasets, raw-label policies, augmentation and single-process loaders."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import albumentations
import cv2
import numpy as np
import tifffile
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from prettyterm import get_logger
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, get_worker_info

from tisam.config.data import ResolvedDataset
from tisam.config.run import TrainConfig
from tisam.data.label_semantics import (
    build_raw_to_global_class_map,
    normalize_policy_raw_mask,
    raw_label_validity,
    remap_mask_to_global_classes,
)
from tisam.data.targets import SUPERVISION_IGNORE_INDEX

logger = get_logger(__name__)

_CLASS_COUNTS_CACHE_FILENAME = "class_counts.json"
_CLASS_COUNTS_CACHE_SCHEMA_VERSION = 2
_SUPPORTED_TILE_SUFFIXES = frozenset({".png", ".tif"})
_SUPPORTED_TILE_SUFFIXES_DISPLAY = ", ".join(sorted(_SUPPORTED_TILE_SUFFIXES))
# Repeated entries intentionally weight common scanner fill colors. Slots 9 and
# 21 invoke continuous neutral sampling, slots 18 and 22 cover saturated white,
# and the final slot retains a warm canvas tail.
_SYNTHETIC_BACKGROUND_BASE_COLOR_SCHEDULE = np.array(
    [
        [232, 232, 232],
        [236, 236, 236],
        [240, 240, 240],
        [230, 230, 230],
        [242, 242, 242],
        [230, 230, 232],
        [232, 232, 232],
        [236, 236, 236],
        [240, 240, 240],
        [237, 237, 237],
        [230, 230, 230],
        [242, 242, 242],
        [232, 232, 232],
        [236, 236, 236],
        [240, 240, 240],
        [230, 230, 232],
        [230, 230, 230],
        [242, 242, 242],
        [255, 255, 255],
        [236, 236, 236],
        [240, 240, 240],
        [237, 237, 237],
        [252, 252, 252],
        [246, 242, 218],
    ],
    dtype=np.float32,
)
_SYNTHETIC_BACKGROUND_CONTINUOUS_NEUTRAL_INDICES = frozenset({9, 21})
_SYNTHETIC_BACKGROUND_SATURATED_WHITE_INDICES = frozenset({18, 22})
_SYNTHETIC_BACKGROUND_JITTERED_WARM_INDICES = frozenset({23})
_SYNTHETIC_BACKGROUND_NEUTRAL_GRAY_RANGE = (232.0, 242.0)
_SYNTHETIC_BACKGROUND_SATURATED_WHITE_VARIATION_SCALE = 0.15
SyntheticBackgroundArtifact = Literal["none", "scanner_border", "dust", "faint_debris", "pen_mark"]
_SYNTHETIC_BACKGROUND_ARTIFACTS: tuple[SyntheticBackgroundArtifact, ...] = (
    "none",
    "scanner_border",
    "dust",
    "faint_debris",
    "pen_mark",
)
_SYNTHETIC_BACKGROUND_ARTIFACT_SCHEDULE: tuple[SyntheticBackgroundArtifact, ...] = (
    "none",
    "faint_debris",
    "scanner_border",
    "none",
    "faint_debris",
    "dust",
    "none",
    "faint_debris",
    "pen_mark",
)


def _resolve_dataset_dir(base_path: Path, primary_name: str, fallback_name: str) -> Path:
    """Prefer the primary directory name, but fall back to the legacy name."""
    primary_dir = base_path / primary_name
    if primary_dir.exists():
        return primary_dir

    fallback_dir = base_path / fallback_name
    if fallback_dir.exists():
        logger.warning(
            f"{primary_dir} does not exist; falling back to legacy directory {fallback_dir}"
        )
        return fallback_dir

    return primary_dir


def _discover_tile_files(
    directory: Path,
    dataset_name: str,
    file_kind: Literal["image", "label"],
) -> dict[str, Path]:
    """Index supported non-recursive tile files by their unique filename stem."""
    if not directory.is_dir():
        return {}

    files_by_stem: dict[str, Path] = {}
    supported_files = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _SUPPORTED_TILE_SUFFIXES
        ),
        key=lambda path: (path.stem, path.suffix.lower(), path.name),
    )
    for path in supported_files:
        existing_path = files_by_stem.get(path.stem)
        if existing_path is not None:
            raise ValueError(
                f"Ambiguous {file_kind} files for dataset '{dataset_name}' and stem "
                f"'{path.stem}': {existing_path} and {path}"
            )
        files_by_stem[path.stem] = path
    return files_by_stem


def _read_tile_file(
    path: Path,
    file_kind: Literal["image", "label"],
) -> np.ndarray:
    """Read one supported image or label tile without changing its pixel values."""
    suffix = path.suffix.lower()
    png_mode: str | None = None
    try:
        if suffix == ".tif":
            array = tifffile.imread(path)
        elif suffix == ".png":
            with Image.open(path) as png:
                if png.format != "PNG":
                    raise ValueError(f"File content is {png.format or 'unidentified'}, not PNG")
                png_mode = png.mode
                array = np.array(png)
        else:
            raise ValueError(
                f"Unsupported tile suffix '{path.suffix}'; expected one of "
                f"{_SUPPORTED_TILE_SUFFIXES_DISPLAY}"
            )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read {file_kind} tile {path}: {exc}") from exc

    if file_kind == "image":
        if suffix == ".png" and png_mode != "RGB":
            raise ValueError(f"Image tile {path} must be an RGB PNG; decoded mode was {png_mode!r}")
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                f"Image tile {path} must have shape HxWx3; decoded shape was {array.shape}"
            )
    elif array.ndim != 2:
        raise ValueError(
            f"Label tile {path} must be single-channel HxW; decoded shape was {array.shape}"
        )
    elif not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_)):
        raise ValueError(
            f"Label tile {path} must contain integer class ids; decoded dtype was {array.dtype}"
        )

    return array


@dataclass(frozen=True)
class TileSample:
    """Image/label pair with dataset metadata needed for canonical remapping."""

    image_file: Path
    label_file: Path
    dataset_idx: int
    dataset_name: str


@dataclass(frozen=True)
class _ClassPixelStatistics:
    """Hold one dataset root's remapped real-label statistics."""

    pixel_counts: np.ndarray
    image_pixel_counts: np.ndarray


def _class_counts_provenance(
    dataset: ResolvedDataset,
    samples: list[TileSample],
    dataset_class_names: list[str],
    raw_to_global_class_idx: np.ndarray,
) -> dict[str, Any]:
    """Describe inputs that determine one root's cached remapped counts."""
    label_manifest = []
    for sample in samples:
        label_stat = sample.label_file.stat()
        label_manifest.append(
            {
                "path": str(sample.label_file.relative_to(dataset.path)),
                "size": label_stat.st_size,
                "mtime_ns": label_stat.st_mtime_ns,
            }
        )
    return {
        "dataset_name": dataset.metadata.name,
        "global_class_names": dataset_class_names,
        "raw_to_global_class_idx": raw_to_global_class_idx.tolist(),
        "supervision_ignored_raw_labels": (dataset.metadata.supervision_ignored_raw_labels),
        "metric_ignored_raw_labels": dataset.metadata.metric_ignored_raw_labels,
        "labels": label_manifest,
    }


def _cached_count_array(value: object, expected_length: int) -> np.ndarray | None:
    """Validate one JSON count array before it can affect loss weights."""
    if not isinstance(value, list) or len(value) != expected_length:
        return None
    if any(type(item) is not int or item < 0 or item > np.iinfo(np.int64).max for item in value):
        return None
    return np.asarray(value, dtype=np.int64)


def _canonical_json(value: object) -> str:
    """Serialize cache data so JSON scalar types participate in comparisons."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _class_counts_cache_content(
    provenance: dict[str, Any],
    pixel_counts: object,
    image_pixel_counts: object,
) -> dict[str, object]:
    """Build the integrity-covered portion of a class-count cache."""
    return {
        "schema_version": _CLASS_COUNTS_CACHE_SCHEMA_VERSION,
        "provenance": provenance,
        "pixel_counts": pixel_counts,
        "image_pixel_counts": image_pixel_counts,
    }


def _class_counts_cache_digest(content: dict[str, object]) -> str:
    """Return the deterministic SHA-256 integrity digest for cache content."""
    return hashlib.sha256(_canonical_json(content).encode()).hexdigest()


def _read_class_counts_cache(
    cache_path: Path,
    provenance: dict[str, Any],
    expected_length: int,
) -> _ClassPixelStatistics | None:
    """Return a validated cache hit or ``None`` for any unusable cache."""
    try:
        with cache_path.open(encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except FileNotFoundError:
        return None
    except (OSError, RecursionError, UnicodeError, ValueError) as exc:
        logger.warning(f"Ignoring unreadable class-count cache {cache_path}: {exc}")
        return None

    if not isinstance(payload, dict):
        logger.warning(f"Ignoring malformed class-count cache {cache_path}.")
        return None
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != _CLASS_COUNTS_CACHE_SCHEMA_VERSION:
        logger.info(f"Class-count cache miss for {cache_path}: schema changed.")
        return None
    if _canonical_json(payload.get("provenance")) != _canonical_json(provenance):
        logger.info(f"Class-count cache miss for {cache_path}: provenance changed.")
        return None

    pixel_counts = _cached_count_array(payload.get("pixel_counts"), expected_length)
    image_pixel_counts = _cached_count_array(
        payload.get("image_pixel_counts"),
        expected_length,
    )
    if (
        pixel_counts is None
        or image_pixel_counts is None
        or not np.any(pixel_counts)
        or not np.any(image_pixel_counts)
        or np.any(pixel_counts > image_pixel_counts)
    ):
        logger.warning(f"Ignoring invalid class-count arrays in {cache_path}.")
        return None

    content = _class_counts_cache_content(
        provenance,
        payload.get("pixel_counts"),
        payload.get("image_pixel_counts"),
    )
    integrity_sha256 = payload.get("integrity_sha256")
    if not isinstance(integrity_sha256, str) or integrity_sha256 != _class_counts_cache_digest(
        content
    ):
        logger.warning(f"Ignoring class-count cache with failed integrity check: {cache_path}.")
        return None

    logger.info(f"Loaded class-count cache from {cache_path}.")
    return _ClassPixelStatistics(pixel_counts, image_pixel_counts)


def _write_class_counts_cache(
    cache_path: Path,
    provenance: dict[str, Any],
    statistics: _ClassPixelStatistics,
) -> None:
    """Atomically persist one root's counts without making training depend on writes."""
    content = _class_counts_cache_content(
        provenance,
        statistics.pixel_counts.tolist(),
        statistics.image_pixel_counts.tolist(),
    )
    payload = dict(content)
    payload["integrity_sha256"] = _class_counts_cache_digest(content)
    temporary = None
    try:
        candidate = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
        descriptor = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        temporary = candidate
        with os.fdopen(descriptor, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, cache_path)
        with suppress(OSError):
            directory = os.open(cache_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        logger.info(f"Wrote class-count cache to {cache_path}.")
    except OSError as exc:
        logger.warning(
            f"Could not write class-count cache {cache_path}; continuing without it: {exc}"
        )
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _compute_class_pixel_statistics(
    samples: list[TileSample],
    raw_to_global_class_idx: np.ndarray,
    class_count: int,
    supervision_ignored_raw_labels: list[int],
) -> _ClassPixelStatistics:
    """Scan one root's supervision-valid labels for weighted-loss setup."""
    pixel_counts = np.zeros(class_count, dtype=np.int64)
    image_pixel_counts = np.zeros(class_count, dtype=np.int64)
    for sample in samples:
        raw_mask = _read_tile_file(sample.label_file, "label")
        mask = remap_mask_to_global_classes(raw_mask, raw_to_global_class_idx)
        policy_raw_mask = normalize_policy_raw_mask(
            raw_mask,
            len(raw_to_global_class_idx),
        )
        supervision_valid = raw_label_validity(
            policy_raw_mask,
            supervision_ignored_raw_labels,
        )
        valid_mask = mask[supervision_valid]
        if valid_mask.size == 0:
            continue
        pixel_counts += np.bincount(valid_mask, minlength=class_count)[:class_count]
        image_pixel_counts[np.unique(valid_mask)] += valid_mask.size
    return _ClassPixelStatistics(pixel_counts, image_pixel_counts)


def _generate_synthetic_background_tile(
    image_hw: tuple[int, int],
    seed: int,
    *,
    artifact: SyntheticBackgroundArtifact = "none",
    base_color_index: int | None = None,
) -> np.ndarray:
    """Generate one deterministic scanner-canvas tile for `TileDataset` training.

    The training dataset calls this for appended synthetic indices so normal and
    data loaders see identical synthetic samples regardless of worker. The
    optional artifact is still background and therefore receives an all-
    `nontissue` mask downstream.
    """
    height, width = image_hw
    rng = np.random.default_rng(seed)
    if base_color_index is None:
        base_color_index = int(rng.integers(len(_SYNTHETIC_BACKGROUND_BASE_COLOR_SCHEDULE)))
    if not 0 <= base_color_index < len(_SYNTHETIC_BACKGROUND_BASE_COLOR_SCHEDULE):
        raise ValueError(f"base_color_index is out of range: {base_color_index}")
    if artifact not in _SYNTHETIC_BACKGROUND_ARTIFACTS:
        raise ValueError(f"Unsupported synthetic background artifact: {artifact}")

    base_color = _sample_synthetic_background_base_color(rng, base_color_index)
    variation_scale = (
        _SYNTHETIC_BACKGROUND_SATURATED_WHITE_VARIATION_SCALE
        if base_color_index in _SYNTHETIC_BACKGROUND_SATURATED_WHITE_INDICES
        else 1.0
    )
    row_axis = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None, None]
    column_axis = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :, None]
    row_gradient = rng.uniform(-1.0, 1.0, size=(1, 1, 3)).astype(np.float32) * variation_scale
    column_gradient = rng.uniform(-1.0, 1.0, size=(1, 1, 3)).astype(np.float32) * variation_scale
    noise_std = float(rng.uniform(0.1, 0.4)) * variation_scale
    luminance_noise = rng.normal(0.0, noise_std, size=(height, width, 1)).astype(np.float32)
    color_noise = rng.normal(
        0.0,
        0.05 * variation_scale,
        size=(height, width, 3),
    ).astype(np.float32)

    image = (
        base_color[None, None, :]
        + row_axis * row_gradient
        + column_axis * column_gradient
        + _generate_low_frequency_illumination(image_hw, rng) * variation_scale
        + luminance_noise
        + color_noise
    )
    image = np.rint(np.clip(image, 0, 255)).astype(np.uint8)

    if artifact == "scanner_border":
        _add_scanner_border(image, rng)
    elif artifact == "dust":
        _add_dust_pecks(image, rng)
    elif artifact == "faint_debris":
        _add_faint_debris(image, rng)
    elif artifact == "pen_mark":
        _add_pen_marks(image, rng)

    return image


def _sample_synthetic_background_base_color(
    rng: np.random.Generator,
    base_color_index: int,
) -> np.ndarray:
    """Sample one scanner-canvas base color for the synthetic tile generator."""
    if base_color_index not in _SYNTHETIC_BACKGROUND_CONTINUOUS_NEUTRAL_INDICES:
        base_color = _SYNTHETIC_BACKGROUND_BASE_COLOR_SCHEDULE[base_color_index]
        if base_color_index not in _SYNTHETIC_BACKGROUND_JITTERED_WARM_INDICES:
            return base_color.copy()
        return base_color + rng.uniform(-2.0, 2.0, size=3).astype(np.float32)

    gray_level = float(rng.uniform(*_SYNTHETIC_BACKGROUND_NEUTRAL_GRAY_RANGE))
    if rng.random() < 0.5:
        return np.full(3, gray_level, dtype=np.float32)

    channel_tint = rng.normal(0.0, 0.75, size=3).astype(np.float32)
    channel_tint -= channel_tint.mean()
    neutral_base = np.full(3, gray_level, dtype=np.float32) + channel_tint
    return np.clip(neutral_base, *_SYNTHETIC_BACKGROUND_NEUTRAL_GRAY_RANGE)


def _generate_low_frequency_illumination(
    image_hw: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate smooth, scanner-agnostic illumination variation for one canvas."""
    height, width = image_hw
    control_height = int(rng.integers(3, 7))
    control_width = int(rng.integers(3, 7))
    control_grid = rng.normal(0.0, 1.0, size=(control_height, control_width)).astype(np.float32)
    illumination = np.asarray(
        cv2.resize(control_grid, (width, height), interpolation=cv2.INTER_CUBIC),
        dtype=np.float32,
    )
    illumination = illumination - np.float32(illumination.mean())
    illumination_std = float(illumination.std())
    if illumination_std > 0:
        illumination = illumination / np.float32(illumination_std)
    illumination = illumination * np.float32(rng.uniform(0.15, 0.75))
    return illumination[:, :, None]


def _add_scanner_border(image: np.ndarray, rng: np.random.Generator) -> None:
    """Add one or two dark scanner-canvas bands to a synthetic background tile."""
    height, width = image.shape[:2]
    edges = rng.choice(4, size=int(rng.integers(1, 3)), replace=False)

    for edge in edges:
        vertical = edge in (0, 1)
        axis_length = width if vertical else height
        thickness = max(1, round(axis_length * float(rng.uniform(0.025, 0.12))))
        border_intensity = float(rng.uniform(25.0, 85.0))
        border_tint = rng.uniform(-4.0, 4.0, size=(1, 1, 3)).astype(np.float32)
        border_color = border_intensity + border_tint

        if edge == 0:
            region = image[:, :thickness]
        elif edge == 1:
            region = image[:, width - thickness :]
        elif edge == 2:
            region = image[:thickness, :]
        else:
            region = image[height - thickness :, :]

        border_noise = rng.normal(0.0, 3.0, size=region.shape).astype(np.float32)
        region[...] = np.rint(np.clip(border_color + border_noise, 0, 255)).astype(np.uint8)


def _add_dust_pecks(image: np.ndarray, rng: np.random.Generator) -> None:
    """Scatter small, dark dust ellipses across a synthetic background tile."""
    height, width = image.shape[:2]
    size_scale = max(0.5, np.sqrt((height * width) / (512 * 512)))
    peck_count = max(6, round(float(rng.integers(12, 31)) * size_scale))

    for _ in range(peck_count):
        center_x = int(rng.integers(width))
        center_y = int(rng.integers(height))
        radius_x = max(1, round(width * float(rng.uniform(0.0015, 0.007))))
        radius_y = max(1, round(height * float(rng.uniform(0.0015, 0.007))))
        x_start = max(0, center_x - radius_x)
        x_stop = min(width, center_x + radius_x + 1)
        y_start = max(0, center_y - radius_y)
        y_stop = min(height, center_y + radius_y + 1)
        y_grid, x_grid = np.ogrid[y_start:y_stop, x_start:x_stop]
        ellipse = ((x_grid - center_x) / radius_x) ** 2 + ((y_grid - center_y) / radius_y) ** 2 <= 1
        region = image[y_start:y_stop, x_start:x_stop]
        opacity = float(rng.uniform(0.4, 0.9))
        dust_color = rng.uniform([25.0, 20.0, 15.0], [115.0, 105.0, 95.0]).astype(np.float32)
        blended = region.astype(np.float32) * (1 - opacity) + dust_color * opacity
        region[ellipse] = np.rint(np.clip(blended[ellipse], 0, 255)).astype(np.uint8)


def _add_faint_debris(image: np.ndarray, rng: np.random.Generator) -> None:
    """Scatter translucent neutral-to-purple debris flakes and rings."""
    height, width = image.shape[:2]
    size_scale = max(0.5, np.sqrt((height * width) / (512 * 512)))
    debris_count = max(8, round(float(rng.integers(16, 41)) * size_scale))

    for _ in range(debris_count):
        center_x = int(rng.integers(width))
        center_y = int(rng.integers(height))
        radius_x = max(1, round(width * float(rng.uniform(0.0015, 0.015))))
        radius_y = max(1, round(height * float(rng.uniform(0.0015, 0.012))))
        x_start = max(0, center_x - radius_x)
        x_stop = min(width, center_x + radius_x + 1)
        y_start = max(0, center_y - radius_y)
        y_stop = min(height, center_y + radius_y + 1)
        y_grid, x_grid = np.ogrid[y_start:y_stop, x_start:x_stop]
        normalized_radius = ((x_grid - center_x) / radius_x) ** 2 + (
            (y_grid - center_y) / radius_y
        ) ** 2
        debris_mask = normalized_radius <= 1
        if rng.random() < 0.4:
            inner_radius = float(rng.uniform(0.35, 0.75))
            debris_mask &= normalized_radius >= inner_radius**2

        region = image[y_start:y_stop, x_start:x_stop]
        opacity = float(rng.uniform(0.08, 0.3))
        debris_color = rng.uniform(
            [135.0, 125.0, 145.0],
            [215.0, 210.0, 220.0],
        ).astype(np.float32)
        blended = region.astype(np.float32) * (1 - opacity) + debris_color * opacity
        region[debris_mask] = np.rint(np.clip(blended[debris_mask], 0, 255)).astype(np.uint8)


def _add_pen_marks(image: np.ndarray, rng: np.random.Generator) -> None:
    """Draw colored freehand-style pen strokes on a synthetic background tile."""
    height, width = image.shape[:2]
    overlay = image.copy()
    palette = np.array(
        [
            [20, 55, 175],
            [20, 120, 75],
            [180, 30, 50],
            [45, 40, 50],
        ],
        dtype=np.uint8,
    )
    stroke_count = int(rng.integers(1, 4))

    for _ in range(stroke_count):
        point_count = int(rng.integers(3, 8))
        points = np.empty((point_count, 2), dtype=np.int32)
        points[0] = [int(rng.integers(width)), int(rng.integers(height))]
        for point_idx in range(1, point_count):
            step = rng.normal([0.0, 0.0], [width * 0.12, height * 0.12])
            points[point_idx] = np.clip(
                points[point_idx - 1] + np.rint(step).astype(np.int32),
                [0, 0],
                [width - 1, height - 1],
            )

        color = tuple(int(value) for value in palette[int(rng.integers(len(palette)))])
        thickness = max(1, round(min(height, width) * float(rng.uniform(0.003, 0.012))))
        cv2.polylines(
            overlay,
            [points],
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    opacity = float(rng.uniform(0.65, 0.95))
    cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, dst=image)


class TileDataset(Dataset):
    """Serve stem-paired PNG/TIFF tiles and optional synthetic background samples.

    `get_tile_loaders` constructs this dataset for training and validation,
    while visualization utilities construct it for real validation tiles only.
    Synthetic indices are appended only for training and always target the
    configured `nontissue` class.
    """

    def __init__(
        self,
        datasets: list[ResolvedDataset],
        dataset_mean: tuple[float, float, float],
        dataset_std: tuple[float, float, float],
        num_classes: int,
        dataset_class_names: list[str],
        target_size: tuple[int, int],
        data_type: str = "train",
        augmentation: albumentations.Compose | None = None,
        apply_class_transform: bool = True,
        synthetic_background_fraction: float = 0.0,
        synthetic_background_seed: int = 0,
        return_metric_validity: bool = False,
    ):
        super().__init__()

        if not 0 <= synthetic_background_fraction < 1:
            raise ValueError("synthetic_background_fraction must be in [0, 1).")
        if synthetic_background_seed < 0:
            raise ValueError("synthetic_background_seed must be non-negative.")
        if synthetic_background_fraction > 0 and data_type != "train":
            raise ValueError("Synthetic background samples are supported only for training.")

        self.datasets = datasets
        self.data_type = data_type
        self.augmentation = augmentation
        self.num_classes = num_classes
        self.dataset_class_names = dataset_class_names
        self.target_size = target_size
        self.apply_class_transform = apply_class_transform
        self.synthetic_background_fraction = synthetic_background_fraction
        self.synthetic_background_seed = synthetic_background_seed
        self.return_metric_validity = return_metric_validity
        self.global_class_to_idx = {
            class_name: class_idx for class_idx, class_name in enumerate(dataset_class_names)
        }
        self.synthetic_background_class_idx = self.global_class_to_idx.get("nontissue")
        if synthetic_background_fraction > 0 and self.synthetic_background_class_idx is None:
            raise ValueError(
                "dataset_class_names must include 'nontissue' when synthetic background "
                "samples are enabled."
            )
        self.raw_to_global_class_idx = [
            build_raw_to_global_class_map(
                dataset.metadata.label_to_class_name, self.global_class_to_idx
            )
            for dataset in datasets
        ]
        ignored_global_classes: set[int] = set()
        supervised_global_classes: set[int] = set()
        for dataset_idx, dataset in enumerate(datasets):
            supervision_ignored = set(dataset.metadata.supervision_ignored_raw_labels)
            for raw_label, global_class_idx in enumerate(self.raw_to_global_class_idx[dataset_idx]):
                if raw_label in supervision_ignored:
                    ignored_global_classes.add(int(global_class_idx))
                else:
                    supervised_global_classes.add(int(global_class_idx))
        self.supervision_ignored_global_class_ids = tuple(
            sorted(ignored_global_classes - supervised_global_classes)
        )
        self.samples: list[TileSample] = []
        self._samples_by_dataset: list[list[TileSample]] = [[] for _ in datasets]
        self._class_pixel_counts: torch.Tensor | None = None
        self._class_image_pixel_counts: torch.Tensor | None = None

        for dataset_idx, dataset in enumerate(datasets):
            images_dir = _resolve_dataset_dir(dataset.path, "images", "im")
            labels_dir = _resolve_dataset_dir(dataset.path, "labels", "label")
            image_files_by_stem = _discover_tile_files(
                images_dir,
                dataset.metadata.name,
                "image",
            )
            if not image_files_by_stem:
                raise FileNotFoundError(
                    f"No supported image files ({_SUPPORTED_TILE_SUFFIXES_DISPLAY}) "
                    f"found in {images_dir}"
                )
            label_files_by_stem = _discover_tile_files(
                labels_dir,
                dataset.metadata.name,
                "label",
            )

            for stem in sorted(image_files_by_stem):
                image_file = image_files_by_stem[stem]
                label_file = label_files_by_stem.get(stem)
                if label_file is None:
                    raise FileNotFoundError(
                        f"Missing supported label file for dataset "
                        f"'{dataset.metadata.name}' and image stem '{stem}' in {labels_dir}"
                    )
                sample = TileSample(
                    image_file=image_file,
                    label_file=label_file,
                    dataset_idx=dataset_idx,
                    dataset_name=dataset.metadata.name,
                )
                self.samples.append(sample)
                self._samples_by_dataset[dataset_idx].append(sample)

        self.real_length = len(self.samples)

        if self.real_length == 0:
            raise FileNotFoundError(
                f"No supported image files ({_SUPPORTED_TILE_SUFFIXES_DISPLAY}) found "
                f"for {data_type} datasets."
            )

        if synthetic_background_fraction > 0:
            synthetic_to_real_ratio = synthetic_background_fraction / (
                1 - synthetic_background_fraction
            )
            self.synthetic_background_count = max(
                1,
                round(self.real_length * synthetic_to_real_ratio),
            )
        else:
            self.synthetic_background_count = 0
        self.length = self.real_length + self.synthetic_background_count

        # Keep the final resize separate so train-only void gating happens after
        # every spatial interpolation but before normalization.
        self.final_spatial_transform = albumentations.Compose(
            [
                albumentations.Resize(target_size[0], target_size[1]),
            ]
        )
        self.tensor_transforms = albumentations.Compose(
            [
                albumentations.Normalize(mean=dataset_mean, std=dataset_std),
                ToTensorV2(),
            ]
        )

        logger.success(
            f"TileDataset initialized: {data_type}, {self.real_length} real and "
            f"{self.synthetic_background_count} synthetic background samples from "
            f"{len(self.datasets)} dataset root(s)"
        )

    def __len__(self):
        return self.length

    def get_sample_dataset_name(self, idx: int) -> str:
        """Return the dataset metadata name associated with a sample index."""
        if idx < 0:
            idx += self.length
        if not 0 <= idx < self.length:
            raise IndexError(idx)
        if idx >= self.real_length:
            return "synthetic_background"
        return self.samples[idx].dataset_name

    def class_pixel_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return real-tile MATLAB-style pixel and image-pixel counts by class.

        Configured TiSAM and DeepLab median-frequency weighted losses call this
        once from the shared training runner. Synthetic background tiles are
        excluded so augmentation does not alter loss weights. As in MATLAB
        `countEachLabel`, each class's image-pixel count is the total size of
        real label tiles in which that class appears.
        """
        if self._class_pixel_counts is not None and self._class_image_pixel_counts is not None:
            return (
                self._class_pixel_counts.clone(),
                self._class_image_pixel_counts.clone(),
            )

        class_count = self.num_classes + 1
        pixel_counts = np.zeros(class_count, dtype=np.int64)
        image_pixel_counts = np.zeros(class_count, dtype=np.int64)
        for dataset_idx, dataset in enumerate(self.datasets):
            samples = self._samples_by_dataset[dataset_idx]
            raw_to_global_class_idx = self.raw_to_global_class_idx[dataset_idx]
            statistics = None
            if self.data_type == "train":
                provenance = _class_counts_provenance(
                    dataset,
                    samples,
                    self.dataset_class_names,
                    raw_to_global_class_idx,
                )
                statistics = _read_class_counts_cache(
                    dataset.path / _CLASS_COUNTS_CACHE_FILENAME,
                    provenance,
                    class_count,
                )
            if statistics is None:
                statistics = _compute_class_pixel_statistics(
                    samples,
                    raw_to_global_class_idx,
                    class_count,
                    dataset.metadata.supervision_ignored_raw_labels,
                )
                if self.data_type == "train":
                    _write_class_counts_cache(
                        dataset.path / _CLASS_COUNTS_CACHE_FILENAME,
                        provenance,
                        statistics,
                    )
            pixel_counts += statistics.pixel_counts
            image_pixel_counts += statistics.image_pixel_counts

        self._class_pixel_counts = torch.from_numpy(pixel_counts)
        self._class_image_pixel_counts = torch.from_numpy(image_pixel_counts)
        return (
            self._class_pixel_counts.clone(),
            self._class_image_pixel_counts.clone(),
        )

    def class_pixel_counts(self) -> torch.Tensor:
        """Return remapped class-pixel counts for compatibility with callers."""
        pixel_counts, _ = self.class_pixel_statistics()
        return pixel_counts

    def __getitem__(self, i):
        """Return one normalized image and one int64 class-index mask."""
        if i < 0:
            i += self.length
        if not 0 <= i < self.length:
            raise IndexError(i)

        enforce_black_void_canvas = False
        if i < self.real_length:
            sample = self.samples[i]
            image = _read_tile_file(sample.image_file, "image")
            raw_mask = _read_tile_file(sample.label_file, "label")
            metadata = self.datasets[sample.dataset_idx].metadata
            raw_label_count = len(metadata.label_to_class_name)
            policy_raw_mask = normalize_policy_raw_mask(
                raw_mask,
                raw_label_count,
            )
            supervision_valid = raw_label_validity(
                policy_raw_mask,
                metadata.supervision_ignored_raw_labels,
            )
            # The shared spatial transform already emits one new array per mask
            # entry, so an alias could not reach the caller. Copying keeps that
            # invariant local to this function instead of resting on that behavior.
            metric_valid = (
                supervision_valid.copy()
                if metadata.metric_ignored_raw_labels == metadata.supervision_ignored_raw_labels
                else raw_label_validity(
                    policy_raw_mask,
                    metadata.metric_ignored_raw_labels,
                )
            )
            mask = remap_mask_to_global_classes(
                raw_mask,
                self.raw_to_global_class_idx[sample.dataset_idx],
            )

            if self.augmentation is not None:
                augmented = self.augmentation(
                    image=image,
                    masks=[mask, supervision_valid, metric_valid],
                )
                image = augmented["image"]
                mask, supervision_valid, metric_valid = augmented["masks"]
            enforce_black_void_canvas = (
                self.data_type == "train" and metadata.train_void_canvas_policy == "enforce_black"
            )
        else:
            assert self.synthetic_background_class_idx is not None
            synthetic_idx = i - self.real_length
            base_color_index = synthetic_idx % len(_SYNTHETIC_BACKGROUND_BASE_COLOR_SCHEDULE)
            artifact_schedule_index = (
                synthetic_idx + synthetic_idx // len(_SYNTHETIC_BACKGROUND_BASE_COLOR_SCHEDULE)
            ) % len(_SYNTHETIC_BACKGROUND_ARTIFACT_SCHEDULE)
            artifact = _SYNTHETIC_BACKGROUND_ARTIFACT_SCHEDULE[artifact_schedule_index]
            image = _generate_synthetic_background_tile(
                self.target_size,
                self.synthetic_background_seed + synthetic_idx,
                artifact=artifact,
                base_color_index=base_color_index,
            )
            mask = np.full(
                self.target_size,
                self.synthetic_background_class_idx,
                dtype=np.int64,
            )
            supervision_valid = np.ones(self.target_size, dtype=bool)
            metric_valid = np.ones(self.target_size, dtype=bool)

        spatially_transformed = self.final_spatial_transform(
            image=image,
            masks=[mask, supervision_valid, metric_valid],
        )
        image = spatially_transformed["image"]
        mask, supervision_valid, metric_valid = spatially_transformed["masks"]
        if enforce_black_void_canvas:
            image[mask == 0] = 0

        # Normalize raw RGB, including enforced black pixels, and convert to tensors.
        transformed = self.tensor_transforms(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].to(dtype=torch.long)
        minimum_value, maximum_value = torch.aminmax(mask)
        minimum_class = int(minimum_value.item())
        maximum_class = int(maximum_value.item())
        if minimum_class < 0 or maximum_class > self.num_classes:
            raise ValueError(
                "Transformed target contains class ids outside the configured range "
                f"[0, {self.num_classes}]: min={minimum_class}, max={maximum_class}."
            )
        supervision_valid_tensor = torch.as_tensor(
            np.asarray(supervision_valid),
            dtype=torch.bool,
        )
        metric_valid_tensor = torch.as_tensor(
            np.asarray(metric_valid),
            dtype=torch.bool,
        )

        # Supervision-ignored pixels carry the shared ignore sentinel instead of
        # the former all-zero one-hot row.
        mask = mask.masked_fill(~supervision_valid_tensor, SUPERVISION_IGNORE_INDEX)

        if self.return_metric_validity:
            return image, mask, metric_valid_tensor
        return image, mask


def get_augmentation(
    image_hw: tuple[int, int],
    crop_scale: tuple[float, float] = (0.08, 1.0),
    color_jitter_p: float = 0.5,
    seed: int | None = None,
) -> tuple[albumentations.Compose, albumentations.Compose]:
    """
    Get data augmentations for training.

    Args:
        image_hw: Target image (height, width) as integers (required)

    Returns:
        Tuple of (train_transform, val_transform)
    """
    h, w = image_hw

    transform_train_fn = albumentations.Compose(
        [
            albumentations.RandomResizedCrop(size=(h, w), scale=crop_scale, p=1.0),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.25),
            albumentations.RandomRotate90(p=0.5),
            albumentations.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=color_jitter_p
            ),
        ],
        seed=seed,
    )

    transform_test_fn = albumentations.Compose(
        [
            albumentations.Resize(height=h, width=w),
        ],
        seed=seed,
    )

    return transform_train_fn, transform_test_fn


def seed_tile_loader_worker(worker_id: int) -> None:
    """Give each DataLoader worker an independent deterministic augmentation stream."""
    del worker_id
    worker_info = get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    augmentation = getattr(dataset, "augmentation", None)
    set_random_seed = getattr(augmentation, "set_random_seed", None)
    if callable(set_random_seed):
        set_random_seed(torch.initial_seed())


class EpochSeededRandomSampler(Sampler[int]):
    """Shuffle one complete dataset from a reproducible seed-plus-epoch stream."""

    def __init__(self, dataset_size: int, *, seed: int, initial_epoch: int = 0) -> None:
        if dataset_size < 0:
            raise ValueError("dataset_size must be non-negative.")
        if initial_epoch < 0:
            raise ValueError("initial_epoch must be non-negative.")
        self.dataset_size = dataset_size
        self.seed = seed
        self.epoch = initial_epoch

    def __iter__(self):
        """Yield the current epoch permutation, then advance the epoch cursor."""
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        self.epoch += 1
        return iter(indices)

    def __len__(self) -> int:
        """Return the number of samples in one epoch."""
        return self.dataset_size


def epoch_aligned_loader_generator(seed: int, initial_epoch: int) -> torch.Generator:
    """Recreate the DataLoader worker-seed stream at one epoch boundary."""
    generator = torch.Generator().manual_seed(seed)
    for _ in range(initial_epoch):
        torch.empty((), dtype=torch.int64).random_(generator=generator)
    return generator


def get_tile_loaders(
    cfg: TrainConfig,
    *,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
    persistent_workers: bool | None = None,
    initial_epoch: int = 0,
):
    """
    Create data loaders for training.

    Args:
        cfg: TrainConfiguration dictionary

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get augmentations
    augs = get_augmentation(
        cfg.model.output_hw,
        crop_scale=cfg.data.crop_scale,
        color_jitter_p=cfg.data.color_jitter_p,
        seed=cfg.seed,
    )

    # Create datasets
    train_dataset = TileDataset(
        datasets=[ResolvedDataset(s.path, s.metadata) for s in cfg.data.train],
        dataset_mean=cfg.data.mean,
        dataset_std=cfg.data.std,
        num_classes=cfg.model.num_classes,
        dataset_class_names=cfg.data.class_names,
        target_size=cfg.model.output_hw,
        data_type="train",
        augmentation=augs[0] if augs else None,
        synthetic_background_fraction=cfg.data.synthetic_background_fraction,
        synthetic_background_seed=cfg.data.synthetic_background_seed,
        return_metric_validity=True,
    )

    val_dataset = TileDataset(
        datasets=[ResolvedDataset(s.path, s.metadata) for s in cfg.data.validation],
        dataset_mean=cfg.data.mean,
        dataset_std=cfg.data.std,
        num_classes=cfg.model.num_classes,
        dataset_class_names=cfg.data.class_names,
        target_size=cfg.model.output_hw,
        data_type="val",
        augmentation=augs[1] if augs else None,
        return_metric_validity=True,
    )

    resolved_pin_memory = pin_memory if pin_memory is not None else "cuda" in cfg.device
    resolved_persistent_workers = (
        persistent_workers
        if persistent_workers is not None
        else cfg.num_workers > 0 and cfg.persistent_workers
    )
    if cfg.num_workers <= 0:
        resolved_persistent_workers = False
    resolved_prefetch_factor = (
        (prefetch_factor if prefetch_factor is not None else 2) if cfg.num_workers > 0 else None
    )
    loader_rank = 0
    train_loader_generator = epoch_aligned_loader_generator(
        cfg.seed + loader_rank,
        initial_epoch,
    )
    val_loader_generator = torch.Generator().manual_seed(cfg.seed + loader_rank + 1_000_000)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=EpochSeededRandomSampler(
            len(train_dataset),
            seed=cfg.seed,
            initial_epoch=initial_epoch,
        ),
        num_workers=cfg.num_workers,
        pin_memory=resolved_pin_memory,
        persistent_workers=resolved_persistent_workers,
        prefetch_factor=resolved_prefetch_factor,
        drop_last=True,
        generator=train_loader_generator,
        worker_init_fn=seed_tile_loader_worker,
    )
    val_dataset_for_loader = val_dataset

    val_loader = DataLoader(
        val_dataset_for_loader,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=resolved_pin_memory,
        persistent_workers=resolved_persistent_workers,
        prefetch_factor=resolved_prefetch_factor,
        drop_last=False,
        generator=val_loader_generator,
        worker_init_fn=seed_tile_loader_worker,
    )

    return train_loader, val_loader

"""Lazy TIFF and RGB PNG tiles consumed by single-process WSI inference."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import albumentations
import numpy as np
import tifffile
import torch
import zarr
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from tisam.data.segmentation_manifest import (
    ResolvedSegmentationRegion,
    resolve_segmentation_region,
)
from tisam.data.tiff_zarr import TiffLevels, open_tiff_levels
from tisam.data.tiling import (
    EFFECTIVE_SIZE,
    TILE_SIZE,
    calculate_patch_positions,
    extract_tile,
)


def read_segmentation_source_shape(path: Path) -> tuple[int, int, int]:
    """Return one supported segmentation source's validated HxWxC shape.

    `tisam.cli.eval.segment` uses this before model loading, while `WSIDataset`
    performs the corresponding pixel read during patch inference.
    """
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            shape = tuple(int(value) for value in tif.series[0].shape)
    elif suffix == ".png":
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(
                    f"Segmentation PNG source must decode as RGB, got "
                    f"format={image.format!r}, mode={image.mode!r}: {path}"
                )
            width, height = image.size
        shape = (height, width, 3)
    else:
        raise ValueError(
            f"Unsupported segmentation source suffix {path.suffix!r}; "
            "expected .tif, .tiff, or .png."
        )
    if len(shape) != 3 or shape[2] != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"Segmentation source must have a positive HxWx3 shape: {path}={shape}.")
    return shape


def open_segmentation_source(
    path: Path,
) -> tuple[TiffLevels | None, zarr.Array | np.ndarray]:
    """Open TIFF lazily or decode one RGB PNG without changing pixel values."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        levels = open_tiff_levels(path)
        return levels, levels.base
    if suffix == ".png":
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(
                    f"Segmentation PNG source must decode as RGB, got "
                    f"format={image.format!r}, mode={image.mode!r}: {path}"
                )
            array = np.asarray(image, dtype=np.uint8).copy()
        return None, array
    raise ValueError(
        f"Unsupported segmentation source suffix {path.suffix!r}; expected .tif, .tiff, or .png."
    )


class WSIDataset(Dataset):
    """Expose WSI tiles to `tisam.inference.segment_wsi`.

    The dataset delegates patch positions and padded tile extraction to
    `tisam.data.tiling`, then returns normalized tensors consumed by the torch
    segmentor inference loop.

    The dataset owns the Zarr store behind `self.wsi` and releases it through
    `close`, or by being used as a context manager. Whoever constructs it owns that
    release, and must perform it only after the `DataLoader` consuming it has shut
    down. Serialization omits open image resources; spawned workers reopen
    the source on their first read and reuse it until they exit. A close in the
    constructing process does not close resources opened by spawned workers.
    There is deliberately no finalizer, because a collector-driven close would
    run at an unpredictable point relative to worker processes.
    """

    def __init__(
        self,
        wsi_path: Path,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        patch_size: int = TILE_SIZE,
        effective_size: int = EFFECTIVE_SIZE,
        region: ResolvedSegmentationRegion | None = None,
        position_indices: Sequence[int] | None = None,
    ):
        self.patch_size = patch_size
        self.effective_size = effective_size

        # TIFF reads stay lazy through Zarr. PNG evaluation ROIs are bounded
        # images and are decoded once per open without color conversion.
        self.wsi_path = Path(wsi_path).resolve()
        self._levels: TiffLevels | None
        self.wsi: zarr.Array | np.ndarray | None
        self._levels, self.wsi = open_segmentation_source(self.wsi_path)

        # Keep fusion positions local to the output mask while WSI reads use
        # source coordinates offset by the resolved region origin.
        h, w, _c = self.wsi.shape
        self.region = region or resolve_segmentation_region(
            None,
            source_shape_hw=(int(h), int(w)),
        )
        if self.region.source_shape_hw != (int(h), int(w)):
            raise ValueError(
                "WSIDataset region source shape does not match the opened WSI: "
                f"{self.region.source_shape_hw} != {(int(h), int(w))}."
            )
        region_h, region_w = self.region.output_shape_hw
        self.full_positions = calculate_patch_positions(
            region_h,
            region_w,
            effective_size=self.effective_size,
        )
        if position_indices is None:
            self.position_indices = np.arange(len(self.full_positions), dtype=np.int64)
        else:
            self.position_indices = np.asarray(position_indices, dtype=np.int64)
            if self.position_indices.ndim != 1:
                raise ValueError(
                    "WSIDataset position_indices must be one-dimensional, got "
                    f"{self.position_indices.shape}."
                )
            if len(self.position_indices) > 0 and (
                int(self.position_indices.min()) < 0
                or int(self.position_indices.max()) >= len(self.full_positions)
            ):
                raise ValueError("WSIDataset position_indices must reference the full patch grid.")
            if len(np.unique(self.position_indices)) != len(self.position_indices):
                raise ValueError("WSIDataset position_indices must not contain duplicates.")
            if len(self.position_indices) > 1 and np.any(
                self.position_indices[1:] <= self.position_indices[:-1]
            ):
                raise ValueError("WSIDataset position_indices must be strictly increasing.")
        self.positions = self.full_positions[self.position_indices]

        # Setup transformation pipeline: normalize and convert to PyTorch tensor
        self.transform = albumentations.Compose(
            [
                albumentations.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        """
        Return the total number of patches in the dataset.

        Returns:
            int: Total number of patches that can be extracted from the WSI
        """
        return len(self.positions)

    def __getstate__(self) -> dict[str, Any]:
        """Serialize configuration without open TIFF stores or decoded images."""
        state = self.__dict__.copy()
        state["_levels"] = None
        state["wsi"] = None
        return state

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.wsi is None:
            self._levels, self.wsi = open_segmentation_source(self.wsi_path)

        # Get the top-left corner position for this patch
        y, x = self.positions[idx]
        left, top, _right, _bottom = self.region.resolved_source_pixel_ltrb

        # Extract tile using the shared utility
        patch = extract_tile(
            self.wsi,
            tile_x=int(x) + left,
            tile_y=int(y) + top,
            tile_size=self.patch_size,
            effective_size=self.effective_size,
            region_ltrb=self.region.resolved_source_pixel_ltrb,
        )

        # Apply normalization and convert to PyTorch tensor
        return self.transform(image=patch)["image"]

    def close(self) -> None:
        """Release the Zarr store backing this dataset.

        `tisam.inference.segment_wsi` and the profiling loader owner call
        this once their `DataLoader` has shut down. Repeat calls are safe.
        """
        if self._levels is not None:
            self._levels.close()
        self._levels = None
        self.wsi = None

    def __enter__(self) -> WSIDataset:
        """Return this dataset for scoped use."""
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Release the store when the scope ends."""
        del exc_type, exc, traceback
        self.close()

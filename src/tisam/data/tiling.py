"""Patch geometry and padded tile extraction shared by WSI inference."""

from __future__ import annotations

import numpy as np
import zarr

EFFECTIVE_SIZE = 800
TILE_SIZE = 1024


def calculate_patch_positions(
    height: int,
    width: int,
    effective_size: int = EFFECTIVE_SIZE,
) -> np.ndarray:
    """Compute a grid of (row, col) tile start positions covering the image.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        effective_size: Stride between tile centers (default EFFECTIVE_SIZE).

    Returns:
        Nx2 integer ndarray of (y, x) positions.
    """
    row_starts = np.arange(0, height, effective_size)
    col_starts = np.arange(0, width, effective_size)
    row_grid, col_grid = np.meshgrid(row_starts, col_starts, indexing="ij")
    return np.stack((row_grid, col_grid), axis=-1).reshape(-1, 2)


def generate_tile_positions_for_region(
    x: int,
    y: int,
    w: int,
    h: int,
    effective_size: int = EFFECTIVE_SIZE,
) -> list[tuple[int, int]]:
    """Offset a patch grid so positions are relative to a subregion origin.

    Args:
        x: Left offset of the subregion within the full image.
        y: Top offset of the subregion within the full image.
        w: Width of the subregion.
        h: Height of the subregion.
        effective_size: Stride between tile centres.

    Returns:
        List of (tile_x, tile_y) tuples in full-image coordinates.
    """
    positions = calculate_patch_positions(h, w, effective_size=effective_size)
    return [(int(px) + x, int(py) + y) for py, px in positions]


def extract_tile(
    wsi: zarr.Array | np.ndarray,
    tile_x: int,
    tile_y: int,
    tile_size: int = TILE_SIZE,
    effective_size: int = EFFECTIVE_SIZE,
    fill_value: int = 255,
    region_ltrb: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Read a tile while padding outside the WSI or an optional source region.

    The tile is read with padding (tile_size - effective_size) // 2 on all
    sides. Pixels outside ``region_ltrb`` are never read and are filled with
    *fill_value* (default white), allowing regional segmentation to avoid
    ingesting neighboring WSI pixels.

    Args:
        wsi: Zarr or NumPy array of shape (H, W, C).
        tile_x: Full-image x coordinate of the tile centre (effective region).
        tile_y: Full-image y coordinate of the tile centre.
        tile_size: Size of the output square tile.
        effective_size: Inner region that is guaranteed valid.
        fill_value: Value used to fill out-of-bounds pixels.
        region_ltrb: Optional half-open source-pixel bounds limiting reads.

    Returns:
        uint8 ndarray of shape (tile_size, tile_size, C).
    """
    ih, iw, c = wsi.shape
    pad = (tile_size - effective_size) // 2
    if region_ltrb is None:
        region_left, region_top, region_right, region_bottom = 0, 0, iw, ih
    else:
        region_left, region_top, region_right, region_bottom = region_ltrb
        if (
            region_left < 0
            or region_top < 0
            or region_right > iw
            or region_bottom > ih
            or region_right <= region_left
            or region_bottom <= region_top
        ):
            raise ValueError(
                f"region_ltrb must be a positive half-open rectangle within "
                f"{iw}x{ih}, got {region_ltrb}."
            )

    y = tile_y - pad
    x = tile_x - pad
    y1, x1 = max(region_top, y), max(region_left, x)
    y2, x2 = min(region_bottom, y + tile_size), min(region_right, x + tile_size)

    patch = np.full((tile_size, tile_size, c), fill_value, dtype=np.uint8)
    patch[y1 - y : y2 - y, x1 - x : x2 - x, :] = wsi[y1:y2, x1:x2, :]
    return patch

"""Closable TIFF pyramid stores consumed by the WSI dataset."""

from __future__ import annotations

import re
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import tifffile
import zarr
from zarr.core.group import Group

__all__ = ["TiffLevels", "open_tiff_levels"]

_LEVEL_KEY_DIGITS = re.compile(r"\d+")


def _level_sort_key(key: str) -> tuple[int, str]:
    """Order pyramid keys numerically where possible, then lexically."""
    digits = _LEVEL_KEY_DIGITS.search(key)
    return (int(digits.group()) if digits else 0, key)


class TiffLevels:
    """Zarr views of one TIFF's pyramid levels, owning the store behind them.

    Closing releases the store, after which the arrays must not be read. Callers
    that hand levels to a longer-lived owner should close through that owner
    rather than letting this resource fall out of scope.
    """

    def __init__(self, levels: tuple[zarr.Array, ...], store: Any) -> None:
        self.levels = levels
        self._store = store

    @property
    def base(self) -> zarr.Array:
        """Return the full-resolution level."""
        return self.levels[0]

    def close(self) -> None:
        """Close the backing store."""
        close = getattr(self._store, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> TiffLevels:
        """Return this resource for scoped use."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the backing store when the scope ends."""
        del exc_type, exc, traceback
        self.close()


def open_tiff_levels(path: Path) -> TiffLevels:
    """Open every pyramid level of one TIFF as Zarr arrays.

    Group keys are ordered numerically and filtered to arrays, so a pyramid whose
    levels are not named `"0"` still resolves. A group with no array levels is an
    error rather than a silently empty result.
    """
    store: Any = tifffile.imread(path, aszarr=True)
    try:
        data = zarr.open(store, mode="r")
        if not isinstance(data, Group):
            return TiffLevels((cast(zarr.Array, data),), store)
        levels = [
            level
            for level in (data[key] for key in sorted(data.keys(), key=_level_sort_key))
            if isinstance(level, zarr.Array)
        ]
        if not levels:
            raise ValueError(f"No image levels found in TIFF zarr store: {path}")
        return TiffLevels(tuple(levels), store)
    except BaseException:
        close = getattr(store, "close", None)
        if callable(close):
            close()
        raise

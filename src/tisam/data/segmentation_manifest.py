"""Source-to-mask coordinate geometry used by WSI inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

NormalizedLTRB = tuple[float, float, float, float]
PixelLTRB = tuple[int, int, int, int]
RegionLTRB = tuple[float, float, float, float]
RegionSpace = Literal["normalized", "pixels"]


@dataclass(frozen=True)
class ResolvedSegmentationRegion:
    """Record one normalized or pixel request on a level-0 source WSI."""

    source_shape_hw: tuple[int, int]
    requested_normalized_ltrb: NormalizedLTRB | None
    resolved_source_pixel_ltrb: PixelLTRB
    requested_pixel_ltrb: PixelLTRB | None = None

    @property
    def requested_region_space(self) -> RegionSpace | None:
        """Return the coordinate space explicitly selected by the caller."""
        if self.requested_pixel_ltrb is not None:
            return "pixels"
        if self.requested_normalized_ltrb is not None:
            return "normalized"
        return None

    @property
    def requested_ltrb(self) -> NormalizedLTRB | PixelLTRB | None:
        """Return the original caller-supplied bounds in their declared space."""
        return self.requested_pixel_ltrb or self.requested_normalized_ltrb

    @property
    def output_shape_hw(self) -> tuple[int, int]:
        """Return the cropped mask height and width."""
        left, top, right, bottom = self.resolved_source_pixel_ltrb
        return bottom - top, right - left

    @property
    def resolved_source_normalized_ltrb(self) -> NormalizedLTRB:
        """Return actual pixel bounds normalized by the source dimensions."""
        height, width = self.source_shape_hw
        left, top, right, bottom = self.resolved_source_pixel_ltrb
        return left / width, top / height, right / width, bottom / height

    def contains_source_pixels(self, requested: PixelLTRB) -> bool:
        """Return whether source-pixel bounds are fully available in this mask."""
        left, top, right, bottom = self.resolved_source_pixel_ltrb
        req_left, req_top, req_right, req_bottom = requested
        return left <= req_left and top <= req_top and req_right <= right and req_bottom <= bottom

    def source_to_local_pixels(self, requested: PixelLTRB) -> PixelLTRB:
        """Translate contained source pixels into this cropped mask's coordinates."""
        if not self.contains_source_pixels(requested):
            raise ValueError(
                f"Requested source region {list(requested)} is outside available "
                f"segmentation region {list(self.resolved_source_pixel_ltrb)}."
            )
        left, top, _, _ = self.resolved_source_pixel_ltrb
        req_left, req_top, req_right, req_bottom = requested
        return (
            req_left - left,
            req_top - top,
            req_right - left,
            req_bottom - top,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON mapping stored for one source WSI."""
        return {
            "mode": (
                "full_slide"
                if self.requested_region_space is None
                else f"{self.requested_region_space}_roi"
            ),
            "requested_region_space": self.requested_region_space,
            "requested_normalized_ltrb": (
                None
                if self.requested_normalized_ltrb is None
                else list(self.requested_normalized_ltrb)
            ),
            "requested_pixel_ltrb": (
                None if self.requested_pixel_ltrb is None else list(self.requested_pixel_ltrb)
            ),
            "resolved_source_pixel_ltrb": list(self.resolved_source_pixel_ltrb),
            "resolved_source_normalized_ltrb": list(self.resolved_source_normalized_ltrb),
            "source_shape_hw": list(self.source_shape_hw),
            "output_shape_hw": list(self.output_shape_hw),
            "bounds": "half-open [left:right, top:bottom]",
            "conversion": (
                "pixel coordinates used directly"
                if self.requested_region_space == "pixels"
                else (
                    "floor(normalized coordinate * source axis length)"
                    if self.requested_region_space == "normalized"
                    else "no region requested; using the full source extent"
                )
            ),
            "output_to_source_translation_xy": [
                self.resolved_source_pixel_ltrb[0],
                self.resolved_source_pixel_ltrb[1],
            ],
        }

    @classmethod
    def from_payload(cls, value: object) -> ResolvedSegmentationRegion:
        """Validate and reconstruct one stored region mapping."""
        if not isinstance(value, Mapping):
            raise ValueError("Segmentation manifest region must be a mapping.")
        source_shape = _two_positive_ints(value.get("source_shape_hw"), "source_shape_hw")
        pixel_ltrb = _four_ints(
            value.get("resolved_source_pixel_ltrb"),
            "resolved_source_pixel_ltrb",
        )
        requested_value = value.get("requested_normalized_ltrb")
        requested_pixel_value = value.get("requested_pixel_ltrb")
        requested_space = value.get("requested_region_space")
        if requested_space is None:
            requested_space = "normalized" if requested_value is not None else None
        if requested_space == "pixels":
            if requested_value is not None:
                raise ValueError(
                    "Pixel-space segmentation manifest region cannot also contain "
                    "normalized request bounds."
                )
            if requested_pixel_value is None:
                raise ValueError(
                    "Pixel-space segmentation manifest region must contain pixel request bounds."
                )
            requested: RegionLTRB | None = cast(
                RegionLTRB,
                _four_ints(requested_pixel_value, "requested_pixel_ltrb"),
            )
        elif requested_space in (None, "normalized"):
            if requested_pixel_value is not None:
                raise ValueError(
                    "Normalized segmentation manifest region cannot also contain "
                    "pixel request bounds."
                )
            if requested_space == "normalized" and requested_value is None:
                raise ValueError(
                    "Normalized segmentation manifest region must contain normalized "
                    "request bounds."
                )
            requested = (
                None
                if requested_value is None
                else _four_floats(requested_value, "requested_normalized_ltrb")
            )
        else:
            raise ValueError(
                "Segmentation manifest `requested_region_space` must be "
                "'normalized', 'pixels', or null."
            )
        resolved = resolve_segmentation_region(
            requested,
            source_shape_hw=source_shape,
            region_space=cast(RegionSpace, requested_space or "normalized"),
        )
        if resolved.resolved_source_pixel_ltrb != pixel_ltrb:
            raise ValueError(
                "Segmentation manifest region pixels do not match its normalized "
                "request and source shape."
            )
        return resolved


def _two_positive_ints(value: object, field: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value)
    ):
        raise ValueError(f"Segmentation manifest `{field}` must contain two positive integers.")
    return int(value[0]), int(value[1])


def _four_ints(value: object, field: str) -> PixelLTRB:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"Segmentation manifest `{field}` must contain four integers.")
    return cast(PixelLTRB, tuple(int(item) for item in value))


def _four_floats(value: object, field: str) -> NormalizedLTRB:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"Segmentation manifest `{field}` must contain four finite numbers.")
    return cast(NormalizedLTRB, tuple(float(item) for item in value))


def resolve_segmentation_region(
    requested: RegionLTRB | None,
    *,
    source_shape_hw: tuple[int, int],
    region_space: RegionSpace = "normalized",
) -> ResolvedSegmentationRegion:
    """Resolve normalized or pixel half-open bounds against a source WSI."""
    source_height, source_width = source_shape_hw
    if source_height <= 0 or source_width <= 0:
        raise ValueError(f"Source WSI shape must be positive, got {source_shape_hw}.")
    if region_space not in ("normalized", "pixels"):
        raise ValueError("Segmentation region space must be either 'normalized' or 'pixels'.")
    if requested is None:
        return ResolvedSegmentationRegion(
            source_shape_hw=source_shape_hw,
            requested_normalized_ltrb=None,
            resolved_source_pixel_ltrb=(0, 0, source_width, source_height),
        )

    left, top, right, bottom = requested
    if not all(math.isfinite(value) for value in requested):
        raise ValueError("Segmentation-region coordinates must all be finite.")
    if right <= left or bottom <= top:
        raise ValueError("Segmentation region must satisfy RIGHT > LEFT and BOTTOM > TOP.")
    if region_space == "normalized":
        if not all(0.0 <= value <= 1.0 for value in requested):
            raise ValueError(
                "Normalized segmentation-region coordinates must all be within [0, 1]."
            )
        normalized_ltrb = cast(NormalizedLTRB, requested)
        requested_pixel_ltrb = None
        pixel_ltrb = (
            int(left * source_width),
            int(top * source_height),
            int(right * source_width),
            int(bottom * source_height),
        )
    else:
        if any(not float(value).is_integer() for value in requested):
            raise ValueError("Pixel segmentation-region coordinates must all be integers.")
        normalized_ltrb = None
        requested_pixel_ltrb = cast(
            PixelLTRB,
            tuple(int(value) for value in requested),
        )
        pixel_ltrb = requested_pixel_ltrb
        pixel_left, pixel_top, pixel_right, pixel_bottom = pixel_ltrb
        if (
            pixel_left < 0
            or pixel_top < 0
            or pixel_right > source_width
            or pixel_bottom > source_height
        ):
            raise ValueError(
                "Pixel segmentation-region coordinates must be within source bounds "
                f"[0, 0, {source_width}, {source_height}]."
            )
    pixel_left, pixel_top, pixel_right, pixel_bottom = pixel_ltrb
    if pixel_right <= pixel_left or pixel_bottom <= pixel_top:
        raise ValueError(
            "Segmentation region resolves to zero pixels on the source WSI; "
            "choose a larger normalized region."
        )
    return ResolvedSegmentationRegion(
        source_shape_hw=source_shape_hw,
        requested_normalized_ltrb=normalized_ltrb,
        resolved_source_pixel_ltrb=pixel_ltrb,
        requested_pixel_ltrb=requested_pixel_ltrb,
    )

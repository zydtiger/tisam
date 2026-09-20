"""SAM3 feature-pyramid geometry shared by TiSAM pixel and query decoders."""

from __future__ import annotations

from dataclasses import dataclass

SpatialShape = tuple[int, int]


@dataclass(frozen=True)
class FeaturePyramidSpec:
    """Describe the four encoder feature levels consumed by Mask2Former."""

    channels: tuple[int, int, int, int]
    spatial_shapes: tuple[SpatialShape, SpatialShape, SpatialShape, SpatialShape]

    def __post_init__(self) -> None:
        if any(channel <= 0 for channel in self.channels):
            raise ValueError("Feature-pyramid channels must be positive.")
        if any(height <= 0 or width <= 0 for height, width in self.spatial_shapes):
            raise ValueError("Feature-pyramid spatial dimensions must be positive.")

    @property
    def top_down_target_shapes(self) -> tuple[SpatialShape, SpatialShape, SpatialShape]:
        """Return the fixed P4, P3, and P2 targets used by top-down fusion."""
        return (
            self.spatial_shapes[2],
            self.spatial_shapes[1],
            self.spatial_shapes[0],
        )


SAM3_FEATURE_PYRAMID_SPEC = FeaturePyramidSpec(
    channels=(256, 256, 256, 256),
    spatial_shapes=((288, 288), (144, 144), (72, 72), (36, 36)),
)

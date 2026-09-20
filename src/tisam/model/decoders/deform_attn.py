"""
Native PyTorch deformable cross-attention module.

Uses ``grid_sample`` so it works without custom CUDA extensions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_


class DeformableCrossAttention(nn.Module):
    """
    Deformable cross-attention for dense spatial fusion.

    Learns a small set of sampling offsets and weights per query location
    instead of attending over every value pixel.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        value_dim: int | None = None,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        self.value_dim = value_dim if value_dim is not None else embed_dim
        self.value_proj: nn.Module
        if self.value_dim != embed_dim:
            self.value_proj = nn.Conv2d(self.value_dim, embed_dim, kernel_size=1)
        else:
            self.value_proj = nn.Identity()

        self.sampling_offsets = nn.Conv2d(embed_dim, num_heads * num_points * 2, kernel_size=1)
        self.attention_weights = nn.Conv2d(embed_dim, num_heads * num_points, kernel_size=1)
        self.output_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self._reset_parameters()

    def _reset_parameters(self):
        assert self.sampling_offsets.bias is not None
        assert self.attention_weights.bias is not None
        assert self.output_proj.bias is not None

        constant_(self.sampling_offsets.weight.data, 0.0)
        constant_(self.sampling_offsets.bias.data, 0.0)
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)

        if isinstance(self.value_proj, nn.Conv2d):
            xavier_uniform_(self.value_proj.weight.data)
            assert self.value_proj.bias is not None
            constant_(self.value_proj.bias.data, 0.0)

        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def _get_reference_points_grid(
        self,
        spatial_shape: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Generate normalized [0, 1] reference points for a spatial shape."""
        height, width = spatial_shape
        y_coords = (torch.arange(0, height, dtype=torch.float32, device=device) + 0.5) / height
        x_coords = (torch.arange(0, width, dtype=torch.float32, device=device) + 0.5) / width
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid = grid_x.new_empty((*grid_x.shape, 2))
        grid[..., 0] = grid_x
        grid[..., 1] = grid_y
        return grid

    def forward(self, query: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for deformable cross-attention.

        Args:
            query: ``(B, C, H_q, W_q)`` tensor used to predict offsets and weights.
            value: ``(B, C_v, H_v, W_v)`` tensor to sample from.

        Returns:
            ``(B, C, H_q, W_q)`` fused output.
        """
        batch_size, _, height_q, width_q = query.shape
        _, _, height_v, width_v = value.shape

        value = self.value_proj(value)
        value = value.view(batch_size, self.num_heads, self.head_dim, height_v, width_v)

        offsets = self.sampling_offsets(query)
        offsets = offsets.view(batch_size, self.num_heads, self.num_points, 2, height_q, width_q)
        offsets = offsets.permute(0, 1, 2, 4, 5, 3)

        scale_tensor = torch.tensor(
            [width_v, height_v],
            dtype=torch.float32,
            device=query.device,
        ).view(1, 1, 1, 1, 1, 2)
        normalized_offsets = offsets / scale_tensor

        weights = self.attention_weights(query)
        weights = weights.view(batch_size, self.num_heads, self.num_points, height_q, width_q)
        weights = torch.nn.functional.softmax(weights, dim=2)

        ref_points = self._get_reference_points_grid((height_q, width_q), query.device)
        ref_points = ref_points.view(1, 1, 1, height_q, width_q, 2)
        sampling_locations = (ref_points + normalized_offsets) * 2.0 - 1.0

        value_input = value.reshape(batch_size * self.num_heads, self.head_dim, height_v, width_v)
        grid_input = sampling_locations.reshape(
            batch_size * self.num_heads,
            self.num_points * height_q,
            width_q,
            2,
        )

        sampled_values = torch.nn.functional.grid_sample(
            value_input,
            grid_input,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_values = sampled_values.view(
            batch_size,
            self.num_heads,
            self.head_dim,
            self.num_points,
            height_q,
            width_q,
        )

        output = (sampled_values * weights.unsqueeze(2)).sum(dim=3)
        output = output.view(batch_size, self.embed_dim, height_q, width_q)
        return self.output_proj(output)

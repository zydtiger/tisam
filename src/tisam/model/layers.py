"""
Mask2Former attention layers used by the decoder stack.

This module depends on PyTorch attention primitives and is called by
`model.decoders.transformer_decoder` to implement masked cross-attention over
pixel-decoder features. In turn, `model.segmentor` reaches this module through
`Mask2FormerDecoder` when SAM3 encoder features and their positional metadata
are consumed by the decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_FUSED_SDPA_HEAD_DIM_MULTIPLE = 8
_FUSED_SDPA_MAX_HEAD_DIM = 256


def build_axial_rope_frequencies(
    height: int,
    width: int,
    head_dim: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build one fixed SAM3-style RoPE grid for decoder cross-attention."""
    base = torch.arange(0, head_dim, 4, device=device, dtype=torch.float32)
    freqs = 1.0 / (10000.0 ** (base[: head_dim // 4] / head_dim))
    positions = torch.arange(height * width, device=device, dtype=torch.float32)
    pos_x = positions.remainder(width)
    pos_y = torch.div(positions, width, rounding_mode="floor")
    freqs_x = torch.outer(pos_x, freqs)
    freqs_y = torch.outer(pos_y, freqs)
    return torch.cat(
        [
            torch.polar(torch.ones_like(freqs_x), freqs_x),
            torch.polar(torch.ones_like(freqs_y), freqs_y),
        ],
        dim=-1,
    )


def get_activation_fn(activation: str) -> nn.Module:
    """Return an activation function instance given a string."""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {activation}")


class MLP(nn.Module):
    """Simple MLP used by decoder-side predictors and attention blocks."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(get_activation_fn(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MaskedMultiScaleAttention(nn.Module):
    """
    Multi-scale cross-attention with mask-based gating and positional keys.

    `Mask2FormerDecoderLayer` in `model.decoders.transformer_decoder` calls this
    module for query-to-image cross-attention. It consumes FPN features from the
    pixel decoder and uses the `use_pos_embed` flag from the decoder stack to
    decide whether image-side keys should receive axial RoPE or learned image
    positions based on the configured positional embedding mode.

    Args:
        d_model: Transformer dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        hard_attn: Whether positive masks should be interpreted as hard gates.
        use_sdpa_attn: Whether to use PyTorch scaled-dot-product attention.

    Forward:
        query: (n, b, d_model) query embeddings
        memory: (b, c, h, w) single-scale feature map
        mask: (b, n, h, w) optional gating weights from previous layer. Soft
            attention expects nonnegative weights; hard attention treats
            positive entries as allowed positions.
        query_pos: optional learned query positions shaped like query
        memory_pos: optional learned image positions shaped like memory

    Returns:
        output: (n, b, d_model) attention output
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        hard_attn: bool = False,
        use_sdpa_attn: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.hard_attn = hard_attn
        self.use_sdpa_attn = use_sdpa_attn
        if self.head_dim % 4 != 0:
            raise ValueError(
                "MaskedMultiScaleAttention requires head_dim divisible by 4 for 2D RoPE, "
                f"got head_dim={self.head_dim}"
            )

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout_p = dropout

    def _apply_axial_rope(
        self,
        tensor: torch.Tensor,
        height: int,
        width: int,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate image-token keys with a precomputed SAM3-style RoPE grid."""
        tensor_complex = torch.view_as_complex(tensor.float().reshape(*tensor.shape[:-1], -1, 2))
        freqs = freqs_cis.view(1, 1, height * width, -1)
        rotated = torch.view_as_real(tensor_complex * freqs).flatten(3)
        return rotated.type_as(tensor)

    def _project_attention_inputs(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_pos: torch.Tensor | None,
        memory_pos: torch.Tensor | None,
        use_pos_embed: bool,
        rope_freqs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
        """Project and reshape attention tensors for `forward`."""
        n, b, d = query.shape
        _, _, h, w = memory.shape
        query_for_q = query if query_pos is None else query + query_pos
        memory_for_k = memory if memory_pos is None else memory + memory_pos
        memory_flat = memory_for_k.flatten(2).permute(2, 0, 1)
        value_flat = memory.flatten(2).permute(2, 0, 1)

        q = self.q_proj(query_for_q)
        k = self.k_proj(memory_flat)
        v = self.v_proj(value_flat)

        q = q.reshape(n, b, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.reshape(h * w, b, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.reshape(h * w, b, self.n_heads, self.head_dim).permute(1, 2, 0, 3)

        if use_pos_embed:
            if rope_freqs is None:
                raise RuntimeError("RoPE mode requires precomputed decoder frequencies.")
            k = self._apply_axial_rope(k, height=h, width=w, freqs_cis=rope_freqs)
        return q, k, v, (n, b, d, h * w)

    def _sdpa_attention_bias(
        self,
        q: torch.Tensor,
        mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        shape: tuple[int, int, int, int],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Build an SDPA-compatible additive bias and zero-output row mask."""
        n, b, _, tokens = shape
        attention_bias: torch.Tensor | None = None
        zero_output_rows: torch.Tensor | None = None

        if attn_mask is not None:
            attention_bias = torch.zeros(
                (1, 1, n, tokens),
                dtype=q.dtype,
                device=q.device,
            )
            attention_bias = attention_bias.masked_fill(
                attn_mask.to(device=q.device).unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )

        if mask is None:
            return attention_bias, zero_output_rows

        mask_flat = mask.to(device=q.device, dtype=q.dtype).flatten(2).unsqueeze(1)
        if self.hard_attn:
            allowed = mask_flat > 0
            all_masked = allowed.sum(dim=-1, keepdim=True) == 0
            safe_allowed = torch.where(all_masked, torch.ones_like(allowed), allowed)
            hard_bias = torch.zeros((b, 1, n, tokens), dtype=q.dtype, device=q.device)
            hard_bias = hard_bias.masked_fill(~safe_allowed, float("-inf"))
            attention_bias = hard_bias if attention_bias is None else attention_bias + hard_bias
            zero_output_rows = all_masked
            return attention_bias, zero_output_rows

        soft_weights = mask_flat.clamp_min(0)
        all_zero = (soft_weights > 0).sum(dim=-1, keepdim=True) == 0
        safe_mask = torch.where(all_zero, torch.ones_like(soft_weights), soft_weights)
        safe_mask = safe_mask.clamp_min(torch.finfo(q.dtype).tiny)
        soft_bias = torch.log(safe_mask).to(dtype=q.dtype)
        attention_bias = soft_bias if attention_bias is None else attention_bias + soft_bias
        zero_output_rows = all_zero
        return attention_bias, zero_output_rows

    def _merge_sdpa_output(
        self,
        sdpa_output: torch.Tensor,
        shape: tuple[int, int, int, int],
        zero_output_rows: torch.Tensor | None,
    ) -> torch.Tensor:
        """Merge SDPA output heads back to query format for `forward`."""
        if zero_output_rows is not None:
            sdpa_output = torch.where(
                zero_output_rows,
                torch.zeros_like(sdpa_output),
                sdpa_output,
            )
        n, b, d, _ = shape
        output = sdpa_output.permute(2, 0, 1, 3).reshape(n, b, d)
        return self.out_proj(output)

    def _augmented_attention_dim(self, num_queries: int) -> int:
        """Return padded head dim for the soft-mask no-bias SDPA path."""
        unpadded_dim = self.head_dim + num_queries
        return (
            (unpadded_dim + _FUSED_SDPA_HEAD_DIM_MULTIPLE - 1)
            // _FUSED_SDPA_HEAD_DIM_MULTIPLE
            * _FUSED_SDPA_HEAD_DIM_MULTIPLE
        )

    def _can_use_augmented_soft_mask_sdpa(
        self,
        q: torch.Tensor,
        mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        shape: tuple[int, int, int, int],
    ) -> bool:
        """Return whether soft decoder masks can be folded into QK scores."""
        n, _, _, _ = shape
        return (
            q.is_cuda
            and mask is not None
            and not self.hard_attn
            and attn_mask is None
            and self._augmented_attention_dim(n) <= _FUSED_SDPA_MAX_HEAD_DIM
        )

    def _augmented_soft_mask_sdpa_output(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        shape: tuple[int, int, int, int],
        dropout_p: float,
    ) -> torch.Tensor:
        """Run soft masked cross-attention as mask-free SDPA for the decoder."""
        n, b, _, tokens = shape
        soft_weights = mask.to(device=q.device, dtype=q.dtype).flatten(2).clamp_min(0)
        all_zero = (soft_weights > 0).sum(dim=-1, keepdim=True) == 0
        safe_mask = torch.where(all_zero, torch.ones_like(soft_weights), soft_weights)
        safe_mask = safe_mask.clamp_min(torch.finfo(q.dtype).tiny)
        soft_bias = torch.log(safe_mask).to(dtype=q.dtype)

        query_selector = torch.eye(n, dtype=q.dtype, device=q.device)
        query_selector = query_selector.view(1, 1, n, n).expand(b, self.n_heads, n, n)
        q_aug = torch.cat((q, query_selector), dim=-1)

        key_bias = (
            soft_bias.transpose(1, 2)
            .unsqueeze(1)
            .expand(
                b,
                self.n_heads,
                tokens,
                n,
            )
        )
        k_aug = torch.cat((k, key_bias / self.scale), dim=-1)

        value_padding = torch.zeros(
            (b, self.n_heads, tokens, n),
            dtype=v.dtype,
            device=v.device,
        )
        v_aug = torch.cat((v, value_padding), dim=-1)

        augmented_dim = self._augmented_attention_dim(n)
        pad_width = augmented_dim - q_aug.shape[-1]
        if pad_width > 0:
            q_aug = torch.nn.functional.pad(q_aug, (0, pad_width))
            k_aug = torch.nn.functional.pad(k_aug, (0, pad_width))
            v_aug = torch.nn.functional.pad(v_aug, (0, pad_width))

        sdpa_output = torch.nn.functional.scaled_dot_product_attention(
            q_aug.contiguous(),
            k_aug.contiguous(),
            v_aug.contiguous(),
            dropout_p=dropout_p,
            scale=self.scale,
        )
        sdpa_output = sdpa_output[..., : self.head_dim]
        zero_output_rows = all_zero.unsqueeze(1)
        return self._merge_sdpa_output(sdpa_output, shape, zero_output_rows)

    def _sdpa_attention_output(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        shape: tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Run cross-attention through PyTorch SDPA."""
        dropout_p = self.dropout_p if self.training else 0.0
        if self._can_use_augmented_soft_mask_sdpa(q, mask, attn_mask, shape):
            assert mask is not None
            return self._augmented_soft_mask_sdpa_output(q, k, v, mask, shape, dropout_p)

        attention_bias, zero_output_rows = self._sdpa_attention_bias(q, mask, attn_mask, shape)
        sdpa_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=dropout_p,
            scale=self.scale,
        )
        return self._merge_sdpa_output(sdpa_output, shape, zero_output_rows)

    def _manual_attention_output(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        shape: tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Run the fallback matmul/softmax attention path when SDPA is disabled."""
        attention_bias, zero_output_rows = self._sdpa_attention_bias(q, mask, attn_mask, shape)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if attention_bias is not None:
            attn_scores = attn_scores + attention_bias

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        if zero_output_rows is not None:
            attn_weights = torch.where(
                zero_output_rows,
                torch.zeros_like(attn_weights),
                attn_weights,
            )
        attn_weights = torch.nn.functional.dropout(
            attn_weights,
            p=self.dropout_p,
            training=self.training,
        )
        attention_output = attn_weights @ v
        return self._merge_sdpa_output(attention_output, shape, None)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        use_pos_embed: bool = False,
        query_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
        rope_freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply masked multi-scale cross-attention.

        Args:
            query: (n, b, d_model) query embeddings
            memory: (b, d_model, h, w) single-scale feature map to attend to
            mask: (b, n, h, w) optional gating weights from previous layer.
                Soft attention expects nonnegative weights; hard attention
                treats positive entries as allowed positions.
            attn_mask: (n, h*w) optional attention mask (True = masked out)
            use_pos_embed: Whether to apply axial RoPE to projected image-side
                keys using the current feature-map height and width.
            query_pos: Optional learned query positions applied before q projection.
            memory_pos: Optional learned image positions applied before k projection.
            rope_freqs: Precomputed axial frequencies for the current FPN level.

        Returns:
            output: (n, b, d_model) attention output
        """
        q, k, v, shape = self._project_attention_inputs(
            query,
            memory,
            query_pos=query_pos,
            memory_pos=memory_pos,
            use_pos_embed=use_pos_embed,
            rope_freqs=rope_freqs,
        )
        if not self.use_sdpa_attn:
            return self._manual_attention_output(q, k, v, mask, attn_mask, shape)
        return self._sdpa_attention_output(q, k, v, mask, attn_mask, shape)

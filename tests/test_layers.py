"""
Unit tests for shared layers.

Tests the core components:
- MLP
- MaskedMultiScaleAttention
"""

import pytest
import torch

from tisam.model.layers import MLP, MaskedMultiScaleAttention


class TestMLP:
    """Tests for MLP layer."""

    def test_output_shape(self):
        """Test that MLP has correct output shape."""
        b, n = 2, 33
        in_dim, hidden_dim, out_dim = 256, 512, 128

        mlp = MLP(in_dim, hidden_dim, out_dim, num_layers=3)
        x = torch.randn(n, b, in_dim)

        output = mlp(x)

        assert output.shape == (n, b, out_dim)

    def test_different_num_layers(self):
        """Test MLP with different number of layers."""
        b, n = 2, 33
        in_dim, hidden_dim, out_dim = 256, 512, 128

        for num_layers in [2, 3]:
            mlp = MLP(in_dim, hidden_dim, out_dim, num_layers=num_layers)
            x = torch.randn(n, b, in_dim)

            output = mlp(x)

            assert output.shape == (n, b, out_dim)

    def test_different_dimensions(self):
        """Test MLP with different input/output dimensions."""
        b, n = 2, 33

        test_cases = [
            (128, 256, 64),
            (256, 512, 256),
            (512, 1024, 128),
        ]

        for in_dim, hidden_dim, out_dim in test_cases:
            mlp = MLP(in_dim, hidden_dim, out_dim, num_layers=3)
            x = torch.randn(n, b, in_dim)

            output = mlp(x)

            assert output.shape == (n, b, out_dim)

    def test_gradient_flow(self):
        """Test that gradients flow through MLP."""
        b, n = 2, 33
        in_dim, hidden_dim, out_dim = 256, 512, 128

        mlp = MLP(in_dim, hidden_dim, out_dim, num_layers=3)
        x = torch.randn(n, b, in_dim, requires_grad=True)

        output = mlp(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_dropout_effect(self):
        """Test that dropout affects output in training mode."""
        b, n = 2, 33
        in_dim, hidden_dim, out_dim = 256, 512, 128

        mlp = MLP(in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5)
        mlp.train()
        x = torch.randn(n, b, in_dim)

        output1 = mlp(x)
        output2 = mlp(x)

        # With dropout, outputs should differ
        assert not torch.allclose(output1, output2)

    def test_dropout_disabled_in_eval(self):
        """Test that dropout is disabled in eval mode."""
        b, n = 2, 33
        in_dim, hidden_dim, out_dim = 256, 512, 128

        mlp = MLP(in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5)
        mlp.eval()
        x = torch.randn(n, b, in_dim)

        output1 = mlp(x)
        output2 = mlp(x)

        # Without dropout, outputs should be identical
        assert torch.allclose(output1, output2)

    def test_activation_relu(self):
        """Test MLP with ReLU activation."""
        b, n = 2, 33
        in_dim, hidden_dim, out_dim = 256, 512, 128

        mlp = MLP(in_dim, hidden_dim, out_dim, num_layers=3, activation="relu")
        x = torch.randn(n, b, in_dim)

        output = mlp(x)

        assert output.shape == (n, b, out_dim)

    def test_activation_gelu(self):
        """Test MLP with GELU activation."""
        b, n = 2, 33
        in_dim, hidden_dim, out_dim = 256, 512, 128

        mlp = MLP(in_dim, hidden_dim, out_dim, num_layers=3, activation="gelu")
        x = torch.randn(n, b, in_dim)

        output = mlp(x)

        assert output.shape == (n, b, out_dim)

    def test_zero_dropout(self):
        """Test MLP with zero dropout."""
        b, n = 2, 33
        in_dim, hidden_dim, out_dim = 256, 512, 128

        mlp = MLP(in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.0)
        mlp.train()
        x = torch.randn(n, b, in_dim)

        output1 = mlp(x)
        output2 = mlp(x)

        # No dropout means deterministic output
        assert torch.allclose(output1, output2)


class TestMaskedMultiScaleAttention:
    """Tests for masked multi-scale attention."""

    def test_output_shape(self):
        """Test that attention has correct output shape."""
        b, n, d = 2, 33, 256
        h, w = 64, 64

        attn = MaskedMultiScaleAttention(d_model=d, n_heads=8)
        query = torch.randn(n, b, d)
        memory = torch.randn(b, d, h, w)

        output = attn(query, memory)

        assert output.shape == (n, b, d)

    def test_attention_with_mask(self):
        """Test that soft mask gating affects attention output."""
        b, n, d = 2, 33, 256
        h, w = 64, 64

        attn = MaskedMultiScaleAttention(d_model=d, n_heads=8, dropout=0.0)
        attn.eval()
        query = torch.randn(n, b, d)
        memory = torch.randn(b, d, h, w)
        mask = torch.sigmoid(torch.randn(b, n, h, w))  # Random mask in [0, 1]

        output_with_mask = attn(query, memory, mask=mask)
        output_no_mask = attn(query, memory, mask=None)

        # Mask should affect attention output.
        assert not torch.allclose(output_with_mask, output_no_mask)

    def test_attention_mask_gating_effect(self):
        """Test that all-zero soft mask rows produce the projected zero output."""
        b, n, d = 2, 33, 256
        h, w = 64, 64

        attn = MaskedMultiScaleAttention(d_model=d, n_heads=8, dropout=0.0)
        attn.eval()
        query = torch.randn(n, b, d)
        memory = torch.randn(b, d, h, w)

        mask = torch.zeros(b, n, h, w)
        output = attn(query, memory, mask=mask)

        assert attn.out_proj.bias is not None
        expected = attn.out_proj.bias.view(1, 1, d).expand(n, b, d)
        assert torch.allclose(output, expected)

    def test_attention_signed_soft_mask_rows(self):
        """Test that signed soft masks are clamped before all-zero detection."""
        b, n, d = 1, 2, 32
        h, w = 4, 4

        attn = MaskedMultiScaleAttention(d_model=d, n_heads=4, dropout=0.0)
        attn.eval()
        query = torch.randn(n, b, d)
        memory = torch.randn(b, d, h, w)

        mask = torch.ones(b, n, h, w)
        mask[:, 0] = -1.0
        mask[:, 1, :, : w // 2] = -1.0

        output = attn(query, memory, mask=mask)

        assert attn.out_proj.bias is not None
        projected_zero = attn.out_proj.bias.view(1, 1, d).expand(n, b, d)
        assert torch.allclose(output[0:1], projected_zero[0:1])
        assert not torch.allclose(output[1:2], projected_zero[1:2])

    def test_attention_with_attn_mask(self):
        """Test attention mask (e.g., padding mask)."""
        b, n, d = 2, 33, 256
        h, w = 64, 64

        attn = MaskedMultiScaleAttention(d_model=d, n_heads=8)
        query = torch.randn(n, b, d)
        memory = torch.randn(b, d, h, w)

        # Mask out some positions
        attn_mask = torch.zeros(n, h * w, dtype=torch.bool)
        attn_mask[:, :100] = True  # Mask first 100 positions

        output = attn(query, memory, attn_mask=attn_mask)
        output_no_mask = attn(query, memory, attn_mask=None)

        assert output.shape == (n, b, d)
        assert not torch.allclose(output, output_no_mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

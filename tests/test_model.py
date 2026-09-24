import pytest
import torch

from tisam import ModelConfig
from tisam.checkpointing import export_weights, load_model
from tisam.checkpointing.weights import apply_weights, checkpoint_state, legacy_model_config


@pytest.mark.parametrize(
    "encoder",
    ["hf-hub:MahmoodLab/UNI2-h", "hf-hub:paige-ai/Virchow2", "hf-hub:MahmoodLab/UNI2-SEAL"],
)
@pytest.mark.parametrize("mode", ["none", "rope", "learned"])
def test_forward_gradient_and_checkpoint(small_model, tmp_path, encoder, mode):
    model = small_model(extra_encoder=encoder, pos_embed_mode=mode)
    x = torch.randn(1, 3, 16, 16)
    logits = model(x)
    assert logits.shape == (1, 3, 16, 16)
    logits.square().mean().backward()
    assert any(p.grad is not None for p in model.mask_decoder.parameters())
    assert all(p.grad is None for p in model.image_encoder.parameters())
    path = tmp_path / "model.safetensors"
    export_weights(model, path)
    # Pretrained encoders have deterministic fixtures, just as a pinned encoder does.
    torch.manual_seed(42)
    restored = load_model(path)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x))
        torch.testing.assert_close(model(x, return_probs=True).sum(1), torch.ones(1, 16, 16))


@pytest.mark.parametrize(
    "deep,cross,early", [(False, True, True), (True, False, True), (True, True, False)]
)
def test_fusion_options(small_model, deep, cross, early):
    model = small_model(
        extra_feature_mode="deep" if deep else "final",
        use_cross_attn=cross,
        early_interpolation=early,
    )
    assert model(torch.randn(1, 3, 16, 16)).shape == (1, 3, 16, 16)


def test_forward_rejects_input_shape_mismatch(small_model):
    model = small_model()
    with pytest.raises(ValueError, match="Expected input tensor"):
        model(torch.randn(1, 3, 8, 8))


def test_input_and_output_resolutions_are_independent(small_model):
    model = small_model(input_hw=(16, 16), output_hw=(8, 8))
    assert model.input_hw == (16, 16) and model.output_hw == (8, 8)
    assert model(torch.randn(1, 3, 16, 16)).shape == (1, 3, 8, 8)


def test_missing_trainable_weights_are_rejected(small_model):
    model = small_model()
    state = checkpoint_state(model)
    state.pop(next(n for n, p in model.named_parameters() if p.requires_grad))
    with pytest.raises(ValueError, match="missing"):
        apply_weights(model, state)


def test_required_config_and_legacy_translation():
    cfg = ModelConfig(num_classes=4, extra_encoder="hf-hub:MahmoodLab/UNI2-h")
    assert cfg.total_queries == 5 and cfg.extra_embed_dim == 1536
    assert cfg.input_hw == (1024, 1024) and cfg.output_hw == (1024, 1024)
    with pytest.raises(ValueError):
        ModelConfig(num_classes=4, extra_encoder=None)
    old = dict(
        dataset_num_classes=4,
        model_extra_encoder=cfg.extra_encoder,
        model_total_queries=5,
        model_num_layers=6,
        model_d_model=256,
        model_use_pos_embed=False,
    )
    assert legacy_model_config(old) == cfg
    with pytest.raises(ValueError, match="dual-encoder"):
        legacy_model_config(old | {"model_use_sam3_encoder": False})


def test_frozen_encoder_parameter_aliases(small_model):
    model = small_model()
    model.image_encoder.alias = model.image_encoder.stem
    # The state writer intentionally stores neither spelling of frozen parameters.
    state = checkpoint_state(model)
    apply_weights(model, state)


def test_legacy_pt_weights(small_model, tmp_path):
    model = small_model()
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": checkpoint_state(model)}, path)
    torch.manual_seed(42)
    restored = load_model(path, config=model.config)
    x = torch.randn(1, 3, 16, 16)
    model.eval()
    torch.testing.assert_close(restored(x), model(x))


def test_reject_changed_sam3_base_weights(tmp_path):
    from tisam.model.encoders.sam3_encoder import verify_sam3_checkpoint

    path = tmp_path / "sam3.pt"
    path.write_bytes(b"not the canonical pretrained weights")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_sam3_checkpoint(path)


def test_frozen_encoders_start_in_eval(small_model):
    model = small_model()
    assert not model.image_encoder.training
    assert not model.extra_encoder.training
    model.train()
    assert not model.image_encoder.training
    assert model.mask_decoder.training


def test_only_exact_historical_projection_keys_are_ignored(small_model):
    model = small_model()
    state = checkpoint_state(model)
    state["mask_decoder.transformer_decoder.pos_enc_proj.0.weight"] = torch.zeros(1)
    apply_weights(model, state)
    state["mask_decoder.transformer_decoder.pos_enc_proj.unknown.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected"):
        apply_weights(model, state)

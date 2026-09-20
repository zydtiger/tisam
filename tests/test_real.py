"""Opt-in CUDA integration with cached encoders and optional external TiSAM weights."""

import gc
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from tisam import ModelConfig, TiSAM, load_model
from tisam.checkpointing import export_weights
from tisam.inference import segment_wsi
from tisam.training.losses import MedianFrequencyWeightedCrossEntropyLoss


@pytest.fixture
def real_cuda(monkeypatch):
    """Keep integration offline and release CUDA allocations between real models."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pretrained integration")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    torch.manual_seed(42)
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.real
@pytest.mark.parametrize(
    "encoder",
    ["hf-hub:MahmoodLab/UNI2-h", "hf-hub:MahmoodLab/UNI2-SEAL", "hf-hub:paige-ai/Virchow2"],
    ids=["uni2", "uni2-seal", "virchow2"],
)
def test_cached_dual_encoder_training_and_wsi(tmp_path, real_cuda, encoder):
    """Exercise real forward, gradients, optimizer update and multi-patch edge assembly."""
    model = TiSAM(ModelConfig(num_classes=2, extra_encoder=encoder, output_hw=(1024, 1024))).cuda()
    optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
    images = torch.randn(1, 3, 1024, 1024, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(images)
        assert logits.shape == (1, 3, 1024, 1024)
        assert torch.isfinite(logits).all()
        loss = MedianFrequencyWeightedCrossEntropyLoss()(
            logits, torch.ones(1, 1024, 1024, dtype=torch.long, device="cuda")
        )["total_loss"]
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    updated = next(p for p in model.parameters() if p.grad is not None and p.grad.count_nonzero())
    before = updated.detach().clone()
    optimizer.step()
    assert not torch.equal(before, updated)
    assert torch.isfinite(loss)
    model.zero_grad(set_to_none=True)
    del optimizer, gradients, logits, loss, images, before, updated
    source = tmp_path / "rgb.tif"
    tifffile.imwrite(
        source,
        np.random.default_rng(42).integers(0, 256, (801, 817, 3), dtype=np.uint8),
        tile=(256, 256),
        compression="zstd",
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        segment_wsi(model, source, tmp_path / "mask.tif", patch_size=1024, effective_size=800)
    mask = tifffile.imread(tmp_path / "mask.tif")
    assert mask.shape == (801, 817)
    assert mask.dtype == np.uint8 and mask.max() < model.total_classes
    assert not (tmp_path / "mask.tif.work").exists()
    with tifffile.TiffFile(tmp_path / "mask.tif") as handle:
        assert handle.pages[0].is_tiled
        assert handle.pages[0].tilewidth == handle.pages[0].tilelength == 256
        assert handle.pages[0].compression.name == "ZSTD"


@pytest.mark.real
def test_external_checkpoint_gpu_round_trip(tmp_path, real_cuda):
    """Import caller-supplied legacy weights and preserve GPU output after native export."""
    checkpoint = os.environ.get("TISAM_TEST_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set TISAM_TEST_CHECKPOINT to an existing TiSAM checkpoint")
    model = load_model(Path(checkpoint), device="cuda")
    images = torch.randn(1, 3, 1024, 1024, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model(images).cpu()
    assert expected.shape == (1, model.total_classes, *model.config.output_hw)
    assert torch.isfinite(expected).all()
    exported = tmp_path / "native.safetensors"
    export_weights(model, exported)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    restored = load_model(exported, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = restored(images).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

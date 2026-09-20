import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from tisam.config import DataConfig, DatasetMetadata, DatasetSource, TrainConfig
from tisam.inference import evaluate, segment_wsi
from tisam.training import runner


def dataset_config(tmp_path, model):
    root = tmp_path / "tiles"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for i in range(2):
        Image.fromarray(np.full((16, 16, 3), 80 + i * 40, dtype=np.uint8)).save(
            root / "images" / f"{i}.png"
        )
        labels = np.full((16, 16), 1 + i, dtype=np.uint8)
        labels[:2] = 0
        Image.fromarray(labels).save(root / "labels" / f"{i}.png")
    source = DatasetSource(
        path=root, metadata=DatasetMetadata(label_to_class_name=["void", "tissue_a", "tissue_b"])
    )
    return TrainConfig(
        model=model.config,
        data=DataConfig(
            class_names=["void", "tissue_a", "tissue_b"],
            train=[source],
            validation=[source],
            test=[source],
            crop_scale=(1, 1),
            color_jitter_p=0,
        ),
        out_dir=tmp_path / "runs",
        epochs=2,
        device="cpu",
        precision="fp32",
        accumulation_steps=1,
        patience=10,
    )


def test_training_resume_and_evaluation(small_model, tmp_path, monkeypatch):
    model = small_model()
    cfg = dataset_config(tmp_path, model)
    monkeypatch.setattr(runner, "TiSAM", lambda config, **kw: small_model(**config.model_dump()))
    result = runner.train(cfg)
    assert result.state.epoch == 1
    checkpoint = runner.latest_checkpoint(cfg.out_dir / cfg.name / "checkpoints")
    assert checkpoint is not None
    resumed = runner.train(cfg)
    assert resumed.state.optimizer_step == result.state.optimizer_step
    metrics = evaluate(model, cfg, split="test")
    assert metrics["split"] == "Test"
    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()
    with pytest.raises(ValueError, match="new run name"):
        runner.train(cfg, resume=False)


def test_wsi_edges_and_resume(small_model, tmp_path, monkeypatch):
    model = small_model()
    source = tmp_path / "rgb.tif"
    tifffile.imwrite(
        source, np.full((19, 23, 3), 100, dtype=np.uint8), tile=(256, 256), compression="zstd"
    )
    output = tmp_path / "mask.tif"
    original = model.forward
    calls = 0

    def fail_once(images, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return original(images, *args, **kwargs)

    monkeypatch.setattr(model, "forward", fail_once)
    with pytest.raises(RuntimeError, match="interrupted"):
        segment_wsi(model, source, output, patch_size=16, effective_size=8)
    assert not output.exists()
    monkeypatch.setattr(model, "forward", original)
    segment_wsi(model, source, output, patch_size=16, effective_size=8)
    mask = tifffile.imread(output)
    assert mask.shape == (19, 23) and mask.dtype == np.uint8
    assert not output.with_name(output.name + ".work").exists()
    with tifffile.TiffFile(output) as tif:
        assert tif.pages[0].is_tiled
        assert tif.pages[0].compression.name == "ZSTD"
    with pytest.raises(FileExistsError):
        segment_wsi(model, source, output, patch_size=16, effective_size=8)


def test_resume_continues_optimizer_and_scheduler(small_model, tmp_path, monkeypatch):
    model = small_model()
    cfg = dataset_config(tmp_path, model)
    monkeypatch.setattr(runner, "TiSAM", lambda config, **kw: small_model(**config.model_dump()))
    original_fit = runner.Trainer.fit

    def first_epoch_only(trainer):
        from dataclasses import replace

        trainer.config = replace(trainer.config, epochs=1)
        return original_fit(trainer)

    monkeypatch.setattr(runner.Trainer, "fit", first_epoch_only)
    first = runner.train(cfg)
    assert first.state.epoch == 0
    monkeypatch.setattr(runner.Trainer, "fit", original_fit)
    second = runner.train(cfg)
    assert second.state.epoch == 1
    assert second.state.optimizer_step == first.state.optimizer_step + 2
    saved = torch.load(
        runner.latest_checkpoint(cfg.out_dir / cfg.name / "checkpoints"), weights_only=True
    )
    assert saved["scheduler"]["last_epoch"] == second.state.optimizer_step
    assert saved["optimizer"]["state"]


def test_wsi_matches_spatial_labels(tmp_path):
    from tisam import ModelConfig

    class SpatialModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = ModelConfig(num_classes=2, extra_encoder="hf-hub:MahmoodLab/UNI2-h")
            self.total_classes = 3

        def forward(self, x):
            return torch.stack((torch.zeros_like(x[:, 0]), x[:, 0], -x[:, 0]), dim=1)

    rgb = np.zeros((19, 23, 3), dtype=np.uint8)
    rgb[::2, ::3, 0] = 255
    source = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(source)
    output = tmp_path / "mask.tif"
    segment_wsi(
        SpatialModel(),
        source,
        output,
        patch_size=16,
        effective_size=8,
        mean=(0.5, 0.5, 0.5),
        std=(1, 1, 1),
    )
    np.testing.assert_array_equal(tifffile.imread(output), np.where(rgb[:, :, 0] > 0, 1, 2))


def test_probability_prediction_under_autocast(small_model):
    from tisam.inference import predict

    model = small_model().eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        probabilities = predict(model, np.full((16, 16, 3), 100, np.uint8), probabilities=True)
    assert probabilities.dtype == np.float32
    np.testing.assert_allclose(probabilities.sum(0), 1, atol=0.005)


def test_training_drops_partial_batch_like_original(small_model, tmp_path):
    from tisam.data.tile_dataset import get_tile_loaders

    model = small_model()
    config = dataset_config(tmp_path, model)
    root = config.data.train[0].path
    (root / "images/2.png").write_bytes((root / "images/1.png").read_bytes())
    (root / "labels/2.png").write_bytes((root / "labels/1.png").read_bytes())
    config.batch_size = 2
    training, validation = get_tile_loaders(config)
    assert len(training) == 1
    assert len(validation) == 2
    assert training.drop_last and not validation.drop_last

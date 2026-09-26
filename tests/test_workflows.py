import hashlib
import json
import logging
import os
import pickle
import stat
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import portalocker
import pytest
import tifffile
import torch
from mammoth.core import claim_logical_run_lease
from mammoth.core.artifacts import ArtifactVerificationError, inspect_artifact
from PIL import Image
from torch.utils.data import DataLoader
from typer.testing import CliRunner

from tisam.cli import app
from tisam.config import DataConfig, DatasetMetadata, DatasetSource, TrainConfig
from tisam.data.tile_dataset import get_tile_loaders
from tisam.data.wsi_dataset import WSIDataset
from tisam.inference import evaluate, segment_wsi
from tisam.training import runner
from tisam.training.checkpoints import TrainingCheckpointPolicy


@pytest.mark.parametrize("command", [[], ["train"], ["eval"], ["eval", "segment"]])
def test_cli_help(command):
    """Build Typer's command tree on every supported Python version."""
    result = CliRunner().invoke(app, [*command, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_train_cli_without_optional_dependencies():
    """Base installs report the training extra before loading a configuration."""
    code = """
import sys
class BlockMammoth:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'mammoth':
            raise ModuleNotFoundError('No module named mammoth', name='mammoth')
sys.meta_path.insert(0, BlockMammoth())
from typer.testing import CliRunner
from tisam.cli import app
result = CliRunner().invoke(app, ['train', 'missing.yaml'])
assert result.exit_code == 2, (result.exit_code, result.exception, result.output)
assert 'tisam[train]' in result.output and 'mammoth' in result.output, result.output
"""
    subprocess.run([sys.executable, "-c", code], check=True)


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


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission and directory fsync contract")
def test_class_count_cache_publication(small_model, tmp_path, monkeypatch):
    """Retain shared-cache permissions and flush both data and its directory entry."""
    cfg = dataset_config(tmp_path, small_model())
    loader, _ = get_tile_loaders(cfg)
    synced_modes = []
    fsync = os.fsync

    def record_sync(descriptor):
        synced_modes.append(os.fstat(descriptor).st_mode)
        fsync(descriptor)

    monkeypatch.setattr(os, "fsync", record_sync)
    previous_umask = os.umask(0o027)
    try:
        counts, _ = loader.dataset.class_pixel_statistics()
    finally:
        os.umask(previous_umask)
    assert counts.sum() == 512
    cache = cfg.data.train[0].path / "class_counts.json"
    assert stat.S_IMODE(cache.stat().st_mode) == 0o640
    assert len(synced_modes) == 2
    assert stat.S_ISREG(synced_modes[0]) and stat.S_ISDIR(synced_modes[1])


def test_class_counts_survive_cache_write_and_cleanup_failure(small_model, tmp_path, monkeypatch):
    """Read-only or failing storage must not discard already computed class counts."""
    cfg = dataset_config(tmp_path, small_model())
    loader, _ = get_tile_loaders(cfg)

    def fail_sync(descriptor):
        raise OSError("cache write failed")

    def fail_cleanup(path, *args, **kwargs):
        raise PermissionError("cache cleanup failed")

    monkeypatch.setattr(os, "fsync", fail_sync)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    counts, _ = loader.dataset.class_pixel_statistics()
    assert counts.sum() == 512


def test_training_resume_and_evaluation(small_model, tmp_path, monkeypatch):
    model = small_model()
    cfg = dataset_config(tmp_path, model)
    monkeypatch.setattr(runner, "TiSAM", lambda config, **kw: small_model(**config.model_dump()))
    result = runner.train(cfg)
    assert result.state.epoch == 1
    checkpoint = runner.latest_checkpoint(cfg.out_dir / cfg.name / "checkpoints")
    assert checkpoint is not None
    attempts = cfg.out_dir / cfg.name / "logs" / "executions"
    first = next(attempts.iterdir())
    records = [json.loads(line) for line in (first / "rank-0.jsonl").read_text().splitlines()]
    assert records[0]["event"] == "process_started"
    assert records[-1]["event"] == "process_completed" and records[-1]["exit_code"] == 0
    progress = [record for record in records if record["event"] == "progress"]
    assert {record["phase"] for record in progress} == {"train", "validation"}
    for record in progress:
        # Short tasks can finish within one clock tick; Mammoth then omits the rate.
        if record.get("throughput") is None:
            assert "batches_per_second" not in record
        else:
            assert record["batches_per_second"] > 0
    summaries = [record for record in records if "epoch_metrics" in record]
    assert len(summaries) == cfg.epochs * 2
    for record in summaries:
        phase = record["phase"]
        expected = (result.training_history if phase == "train" else result.validation_history)[
            record["epoch"]
        ]
        assert record["epoch_metrics"] == {
            f"{phase}/{name}": value for name, value in expected.items()
        }
    publications = [record for record in records if "checkpoints" in record]
    assert len(publications) == cfg.epochs
    latest = next(item for item in publications[-1]["checkpoints"] if item["role"] == "latest")
    assert latest["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert latest["size_bytes"] == checkpoint.stat().st_size
    assert publications[-1]["retired"]
    first_metadata = json.loads((first / "execution.json").read_text())
    assert Path(first_metadata["config_reference"]) == (first / "config.json").resolve()
    assert (first / "rank-0.log").exists()
    assert list((first / "tensorboard").glob("events.*"))
    resumed = runner.train(cfg)
    assert resumed.state.optimizer_step == result.state.optimizer_step
    second = next(path for path in attempts.iterdir() if path != first)
    metadata = json.loads((second / "execution.json").read_text())
    assert metadata["previous_execution_id"] == first.name
    assert metadata["resume_checkpoint_sha256"] == latest["sha256"]
    assert metadata["starting_global_step"] == result.state.global_step
    assert metadata["starting_epoch"] == cfg.epochs
    assert json.loads((second / "rank-0.jsonl").read_text().splitlines()[-1])["exit_code"] == 0
    metrics = evaluate(model, cfg, split="test")
    assert metrics["split"] == "Test"
    assert {"accuracy", "precision", "recall", "f1"} <= metrics.keys()
    with pytest.raises(ValueError, match="new run name"):
        runner.train(cfg, resume=False)
    initialized = cfg.model_copy(update={"name": "initialized", "epochs": 1})
    best = checkpoint.parent / "best.safetensors"
    runner.train(initialized, initialize_from=best)
    initialization_log = next(
        (initialized.out_dir / initialized.name / "logs" / "executions").glob("*/rank-0.jsonl")
    )
    initialization_records = [
        json.loads(line) for line in initialization_log.read_text().splitlines()
    ]
    source = next(
        record
        for record in initialization_records
        if record.get("task_id") == "initialization-source"
    )
    assert source["artifacts"][0]["sha256"] == hashlib.sha256(best.read_bytes()).hexdigest()


@pytest.mark.parametrize("interrupted", [False, True])
def test_training_setup_failure_is_logged_and_unlocks(
    small_model, tmp_path, monkeypatch, interrupted
):
    """Setup failures close logs and allow another attempt without stale ownership."""
    cfg = dataset_config(tmp_path, small_model())
    root = logging.getLogger()
    original_handlers, original_level = list(root.handlers), root.level

    def fail_setup(*args, **kwargs):
        with pytest.raises(RuntimeError, match="already active"):
            claim_logical_run_lease(cfg.out_dir / cfg.name)
        if interrupted:
            raise KeyboardInterrupt
        raise RuntimeError("deliberate loader failure")

    monkeypatch.setattr(runner, "get_tile_loaders", fail_setup)
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt if interrupted else RuntimeError):
            runner.train(cfg)
        assert root.handlers == original_handlers
        assert root.level == original_level
    attempts = list((cfg.out_dir / cfg.name / "logs" / "executions").iterdir())
    assert len(attempts) == 2
    for attempt in attempts:
        records = [json.loads(line) for line in (attempt / "rank-0.jsonl").read_text().splitlines()]
        assert any(record["event"] == "phase_failed" for record in records)
        assert records[-1]["event"] == "process_completed"
        assert records[-1]["exit_code"] == (130 if interrupted else 1)
        message = "Training interrupted" if interrupted else "deliberate loader failure"
        assert message in (attempt / "rank-0.log").read_text()


def test_training_run_lock_prevents_concurrent_attempt(small_model, tmp_path):
    """A competing producer must fail before creating logs or touching checkpoints."""
    cfg = dataset_config(tmp_path, small_model())
    directory = cfg.out_dir / cfg.name
    directory.mkdir(parents=True)
    with claim_logical_run_lease(directory):
        with pytest.raises(RuntimeError, match="already active"):
            runner.train(cfg)
    assert not (directory / "logs" / "executions").exists()
    assert not (directory / "checkpoints").exists()


@pytest.mark.parametrize("failure_stage", ["preflight", "logging"])
def test_training_early_failure_releases_run_ownership(
    small_model, tmp_path, monkeypatch, failure_stage
):
    """Failures before the session starts must not strand the run or logger."""
    from tisam.training import observability

    cfg = dataset_config(tmp_path, small_model())
    root = logging.getLogger()
    original_handlers, original_level = list(root.handlers), root.level
    kwargs = {}
    if failure_stage == "preflight":
        checkpoint = tmp_path / "historical.pt"
        torch.save({"weights": {}}, checkpoint)
        kwargs["checkpoint"] = checkpoint
        expected_error, message = ValueError, "Use initialize_from"
    else:

        def fail_tensorboard(*args, **kwargs):
            raise RuntimeError("deliberate logging setup failure")

        monkeypatch.setattr(observability, "TensorBoardSink", fail_tensorboard)
        expected_error, message = RuntimeError, "deliberate logging setup failure"
    for _ in range(2):
        with pytest.raises(expected_error, match=message):
            runner.train(cfg, **kwargs)
        with claim_logical_run_lease(cfg.out_dir / cfg.name):
            pass
        assert root.handlers == original_handlers
        assert root.level == original_level


def test_resume_rejects_checkpoint_changed_after_preflight(small_model, tmp_path):
    """Do not restore state from bytes different from the recorded resume source."""
    model = small_model()
    cfg = dataset_config(tmp_path, model)
    policy = TrainingCheckpointPolicy(model, None, None, None, cfg)
    path = tmp_path / "resume.pt"
    torch.save({"training_schema": 1, "training_config": cfg.model_dump(mode="json")}, path)
    policy.resume_receipt = inspect_artifact(path)
    torch.save({"different": "checkpoint"}, path)
    with pytest.raises(ArtifactVerificationError, match="changed after attempt creation"):
        policy.read(path)


@pytest.mark.parametrize("suffix", [".tif", ".png"])
def test_wsi_spawn_matches_local_reads(tmp_path, suffix):
    """Spawn workers reopen images while serialization leaves the parent usable."""
    source = tmp_path / ("rgb" + suffix)
    pixels = np.arange(19 * 23 * 3, dtype=np.uint16).reshape(19, 23, 3).astype(np.uint8)
    if suffix == ".tif":
        tifffile.imwrite(source, pixels, tile=(256, 256), compression="zstd")
    else:
        Image.fromarray(pixels).save(source)
    with WSIDataset(source, (0, 0, 0), (1, 1, 1), 16, 8) as dataset:
        expected = torch.stack([dataset[i] for i in range(len(dataset))])
        with pickle.loads(pickle.dumps(dataset)) as restored:
            torch.testing.assert_close(restored[0], expected[0], rtol=0, atol=0)
        loader = DataLoader(dataset, batch_size=2, num_workers=1, multiprocessing_context="spawn")
        actual = torch.cat(list(loader))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(dataset[0], expected[0], rtol=0, atol=0)
    dataset.close()
    source.unlink()


@pytest.mark.parametrize("num_workers", [0, 1])
def test_wsi_edges_and_resume(small_model, tmp_path, monkeypatch, num_workers):
    monkeypatch.setattr(
        "tisam.inference.wsi.DataLoader",
        partial(DataLoader, multiprocessing_context="spawn" if num_workers else None),
    )
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
        segment_wsi(model, source, output, patch_size=16, effective_size=8, num_workers=num_workers)
    assert not output.exists()
    monkeypatch.setattr(model, "forward", original)
    segment_wsi(model, source, output, patch_size=16, effective_size=8, num_workers=num_workers)
    mask = tifffile.imread(output)
    assert mask.shape == (19, 23) and mask.dtype == np.uint8
    assert not output.with_name(output.name + ".work").exists()
    with tifffile.TiffFile(output) as tif:
        assert tif.pages[0].is_tiled
        assert tif.pages[0].compression.name == "ZSTD"
    with pytest.raises(FileExistsError):
        segment_wsi(model, source, output, patch_size=16, effective_size=8, num_workers=num_workers)


def test_wsi_writer_lock_excludes_other_processes(small_model, tmp_path, monkeypatch):
    """Keep scratch exclusive during inference and release it after interruption."""
    model = small_model()
    source = tmp_path / "rgb.png"
    Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8)).save(source)
    output = tmp_path / "mask.tif"
    lock = tmp_path / "mask.tif.work" / "writer.lock"

    def check_lock_then_interrupt(images):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import portalocker, sys\n"
                "try:\n"
                "    with portalocker.Lock(sys.argv[1], mode='a+b', timeout=0): pass\n"
                "except portalocker.exceptions.LockException:\n"
                "    sys.exit(3)\n",
                str(lock),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 3, result.stderr
        raise RuntimeError("interrupted with lock held")

    monkeypatch.setattr(model, "forward", check_lock_then_interrupt)
    with pytest.raises(RuntimeError, match="interrupted with lock held"):
        segment_wsi(model, source, output, patch_size=16, effective_size=8)
    with portalocker.Lock(lock, mode="a+b", timeout=0):
        assert not output.exists()


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

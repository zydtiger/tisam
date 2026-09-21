"""Console entry point with optional workflows loaded only when invoked."""

import importlib
import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from tisam.checkpointing.weights import load_model
from tisam.config.run import load_config

app = typer.Typer(no_args_is_help=True, help="Dual-encoder TiSAM training and inference.")
eval_app = typer.Typer(no_args_is_help=True)
app.add_typer(eval_app, name="eval")


def workflow(module: str, extra: str):
    """Report the required extra without importing it for unrelated commands."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        raise typer.BadParameter(
            f"This command requires tisam[{extra}]; missing {error.name}"
        ) from error


@app.command("train")
def train_command(
    config: Path,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    checkpoint: Optional[Path] = None,
    initialize_from: Optional[Path] = None,
    trusted: bool = False,
):
    """Train one configuration; resume its latest checkpoint by default."""
    workflow("tisam.training.runner", "train").train(
        load_config(config),
        resume=resume,
        checkpoint=checkpoint,
        initialize_from=initialize_from,
        trusted=trusted,
    )


def run_evaluation(config: Path, checkpoint: Path, output: Path, split: str, trusted: bool):
    """Share config and model loading between validation and test commands."""
    cfg = load_config(config)
    module = workflow("tisam.inference", "inference")
    model = load_model(
        checkpoint,
        config=cfg.model,
        device=cfg.device,
        sam3_checkpoint=cfg.sam3_checkpoint,
        trusted=trusted,
    )
    result = module.evaluate(model, cfg, split=split)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    typer.echo(json.dumps(result, indent=2))


@eval_app.command("validate")
def validate_command(
    config: Path, checkpoint: Path, output: Path = Path("validation.json"), trusted: bool = False
):
    """Evaluate the configured validation tiles and write foreground metrics."""
    run_evaluation(config, checkpoint, output, "validation", trusted)


@eval_app.command("test")
def test_command(
    config: Path, checkpoint: Path, output: Path = Path("test.json"), trusted: bool = False
):
    """Evaluate the configured held-out test tiles."""
    run_evaluation(config, checkpoint, output, "test", trusted)


@eval_app.command("segment")
def segment_command(
    config: Path,
    checkpoint: Path,
    source: Path,
    output: Path,
    patch_size: int = 1024,
    effective_size: int = 800,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    trusted: bool = False,
):
    """Segment one RGB TIFF/PNG into a tiled TIFF mask with resumable scratch."""
    cfg = load_config(config)
    module = workflow("tisam.inference", "inference")
    model = load_model(
        checkpoint,
        config=cfg.model,
        device=cfg.device,
        sam3_checkpoint=cfg.sam3_checkpoint,
        trusted=trusted,
    )
    result = module.segment_wsi(
        model,
        source,
        output,
        patch_size=patch_size,
        effective_size=effective_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        mean=cfg.data.mean,
        std=cfg.data.std,
        resume=resume,
    )
    typer.echo(str(result))

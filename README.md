# TiSAM

TiSAM combines the SAM3 image encoder with a pathology foundation encoder and a
semantic pixel/query decoder. This project provides the architecture as a Python
module, single-process training, validation, and tile/whole-slide inference.
It contains no datasets or pretrained weights. Client/server applications and
other segmentation architectures are outside this package.

Source is available at [zydtiger/tisam](https://github.com/zydtiger/tisam).
No package registry release or project license has been selected yet. Release
automation is not configured.

## Install

Model loading, training, prediction, and evaluation support Python 3.9 or newer,
PyTorch 2.8 or newer, and NumPy 1.26 or newer. Linux CI covers representative versions 3.9, 3.12, and 3.14;
the Python 3.9 job also checks PyTorch 2.8.0, torchvision 0.23.0, NumPy 1.26.4,
tifffile 2024.8.28, and Zarr 2.18.2. Newer Python versions retain the modern
dependency stack.
CUDA is needed for practical pretrained-model execution. This project does not
select a CUDA backend in its package metadata; select one with uv when installing
as described below.

Python 3.9–3.10 use tifffile before 2025.5.21 with Zarr 2; Python 3.11+ use
tifffile 2025.5.21+ with Zarr 3. These pairs preserve lazy TIFF reads. WSI writer
locks use portalocker rather than a Unix-only API; Windows execution still needs
platform validation beyond the Linux test matrix.

Install directly from GitHub into an existing environment:

```sh
uv pip install 'tisam @ git+https://github.com/zydtiger/tisam.git'
uv pip install 'tisam[train,inference] @ git+https://github.com/zydtiger/tisam.git'
```

For development, from the project root:

```sh
uv sync --all-extras --group dev
uv run --no-sync tisam --help
```

For an external environment, install the project root:

```sh
uv pip install .
uv pip install '.[train,inference]'
```

The base package provides `TiSAM`, `ModelConfig`, and checkpoint loading.
`train` adds Mammoth, augmentation, datasets and TensorBoard logging;
`inference` adds RGB image/TIFF/WSI workflows. Optional workflows are not
imported by `import tisam` or CLI help. SAM3, PrettyTerm, and Mammoth use immutable public Git
references in the package metadata, so installation does not depend on uv source
overrides or a second local checkout. Git is required to resolve those sources.

### Select a PyTorch backend

Use `uv pip install --torch-backend` to select the PyTorch wheel index, including
when TiSAM is a dependency of a downstream package. These examples use uv 0.10.10
and an existing environment with the desired Python version. For a typical
installation on the target GPU machine with Python 3.10+, start with `auto`.
Use `cu128` for Python 3.9, and an explicit backend for CI or deployments that
require a fixed backend:

```sh
# Recommended for installation on the target GPU machine with Python 3.10+.
uv pip install --torch-backend=auto 'tisam[train,inference] @ git+https://github.com/zydtiger/tisam.git'

# CUDA 13.0: requires a compatible Python 3.10+ environment.
uv pip install --torch-backend=cu130 'tisam[train,inference] @ git+https://github.com/zydtiger/tisam.git'

# CUDA 12.8: also supports Python 3.9 with PyTorch 2.8.0 / torchvision 0.23.0.
uv pip install --torch-backend=cu128 'tisam[train,inference] @ git+https://github.com/zydtiger/tisam.git'

# The same option works when installing from the project root.
uv pip install --torch-backend=auto '.[train,inference]'
```

Omit `[train,inference]` for the base package. Use a fresh environment for a new
backend, or add `--reinstall-package torch --reinstall-package torchvision` when
switching an existing installation. Add `--dry-run` to preview dependency
resolution without installing packages. The environment variable form is
equivalent, for example `UV_TORCH_BACKEND=cu128 uv pip install '.[train,inference]'`.

Dependency resolution follows these rules:

- Without `--torch-backend` or `UV_TORCH_BACKEND`, uv uses the configured indexes
  (normally PyPI). TiSAM does not guarantee a default CUDA version or detect the
  GPU. `cu12` and `cu13` are not TiSAM extras.
- An explicit backend selects the index for PyTorch packages, including
  transitive `torch` and `torchvision` dependencies. Other dependencies continue
  to use the configured indexes. All Python, platform, package, and selected-extra
  constraints still apply; the backend does not pin a PyTorch version.
- Python 3.9 with `cu128` resolves to PyTorch 2.8.0 and torchvision 0.23.0.
  Python 3.9 with `cu130` fails because compatible `cp39` wheels are unavailable;
  uv does not fall back to another backend or upgrade Python. A downstream
  `requires-python = ">=3.9"` is a minimum requirement, not a pin to Python 3.9.
- `--torch-backend=auto` detects the local GPU/driver and selects a backend, with
  CPU fallback when no supported GPU is detected. It then resolves dependencies
  within that backend; it does not switch backends if the Python or dependency
  constraints cannot be satisfied. For example, Python 3.9 still fails if `auto`
  selects `cu130`; specify `cu128` instead. Successful resolution alone does not
  verify that the GPU and driver can execute the selected build.
- In uv 0.10.10, this option is available through the `uv pip` interface, not
  `uv sync`. `uv pip install` resolves the package metadata without using this
  repo's `uv.lock`; it does not update a downstream lockfile. The development
  `uv sync` command above uses the locked dependencies. A later `uv sync` or
  synchronizing `uv run` can replace a backend installed with `uv pip`; use
  `uv run --no-sync` to run that environment without synchronization.

See the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
for backend selection and project-level index configuration.

### Pretrained weights

Pretrained SAM3, UNI2-h, Virchow2 and UNI2-SEAL weights remain external. Obtain
access from their respective providers and authenticate with `hf auth login`
where required. Weights download only when constructing an encoder, not when
installing or importing TiSAM. The optional `sam3_checkpoint` path must contain the same canonical SAM3 weights.
Existing Hugging Face cache entries are reused;
`HF_HUB_OFFLINE=1` prevents network access. Encoder code and weights retain their
own license/access conditions; this project does not grant additional rights.

## Use the architecture

```python
import torch
from tisam import TiSAM, ModelConfig

config = ModelConfig(
    num_classes=2,  # tissue classes, excluding canonical void at index 0
    extra_encoder="hf-hub:MahmoodLab/UNI2-h",
)
model = TiSAM(config).to("cuda").eval()
images = torch.randn(1, 3, 1024, 1024, device="cuda")  # normalized RGB
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(images)  # B x (num_classes + 1) x H x W
    labels = logits.argmax(dim=1)
```

Set `extra_encoder` to `hf-hub:paige-ai/Virchow2` or
`hf-hub:MahmoodLab/UNI2-SEAL` for the other supported foundation encoders.
Both encoders are always present. `total_queries` defaults to `num_classes + 1`.
The foundation width is derived from its identity. `input_hw` and `output_hw`
both default to `(1024, 1024)`, and `model(images)` requires an input whose
spatial shape matches `input_hw`. Defaults use final-feature fusion, no
positional embedding, no SAM projection, frozen encoders and six query-decoder
layers. `ModelConfig` exposes the existing fusion, positional embedding,
projection, interpolation and finetuning options.

## Train and evaluate

Start with [the example configuration](examples/train.yaml). Each dataset root
contains paired `images/<stem>.png` and `labels/<stem>.png` files (TIFF `.tif`
pairs are also supported). Images are RGB and masks contain integer raw IDs.
`metadata.label_to_class_name` maps each raw ID to a canonical class name.
Canonical `class_names` begins with `void`; preserve tissue identifiers exactly.
Input paths resolve relative to the configuration file. Output paths resolve
relative to the working directory and default to `./runs`.

```sh
uv run --no-sync tisam train examples/train.yaml
uv run --no-sync tisam eval validate examples/train.yaml runs/tisam/checkpoints/best.safetensors
uv run --no-sync tisam eval test examples/train.yaml runs/tisam/checkpoints/best.safetensors
```

Training resumes its latest checkpoint by default. Use a new run name for fresh
training or `--initialize-from path/to/weights.safetensors` to initialize a new
run. The default objective is median-frequency weighted cross-entropy, adopted
from the CODA DeepLab implementation. Every supervised class needs training
support; explicitly mark unsupervised raw IDs in dataset metadata. Focal/Dice
and tempered weighted-CE/Dice/background objectives are available as explicit
options. The latter requires a canonical `background` tissue class distinct
from `void`.

Training is single-process and supports AMP, gradient accumulation, separate
encoder learning rates, early stopping, optional `torch.compile`, resumable
`.pt` checkpoints and best `.safetensors` weights. Every invocation creates an
attempt under `runs/<name>/logs/executions/<execution-id>/` containing:

- `rank-0.jsonl`: Mammoth lifecycle, progress, epoch metrics, and checkpoint
  publication receipts (paths, roles, epochs, byte sizes, SHA-256 hashes, and
  retired paths).
- `rank-0.log`: Python logging diagnostics, including setup failures.
- `execution.json` and `config.json`: attempt identity, resume provenance, and
  the resolved configuration snapshot.
- `tensorboard/`: dense metric history. Point TensorBoard at `runs/<name>/logs`
  to include all attempts and older runs.

JSONL progress records include `batches_per_second`, an epoch-to-date rate.
Mammoth's native `throughput` counts accumulation windows/s during training and
batches/s during validation; `throughput_unit` identifies which. Batch rates use
the actual consumed batch count, including a shorter final accumulation window.
With batch size one, batches/s also equals tiles/s. Epoch summaries retain their
full metric mapping as `epoch_metrics`.

The convenience `runs/<name>/config.json` contains the latest attempt's config;
older attempt snapshots remain intact. A portable file lock prevents two TiSAM
training invocations from writing the same run concurrently. New logging takes
effect on the next invocation; a running process keeps its original logging setup.
Use the same training configuration and epoch horizon when resuming. Weight
initialization is the supported way to change the training objective or dataset.
See [the checkpoint contract](docs/architecture.md) for details.

Validation and test report foreground accuracy, class-macro precision, recall
and F1 plus a confusion matrix. They apply raw-label metric validity before
canonical remapping. Mean foreground IoU remains available internally for
training checkpoint selection; it is not part of the public evaluation summary.

## Predict images and whole slides

```python
from tisam import load_model
from tisam.inference import predict, segment_wsi

model = load_model("weights.safetensors", device="cuda")
# labels = predict(model, rgb_uint8_array)
segment_wsi(model, "slide.tif", "mask.tif")
```

```sh
uv run --no-sync tisam eval segment examples/train.yaml weights.safetensors slide.tif mask.tif
```

`predict` accepts a uint8 HWC RGB image and returns HW labels; request
`probabilities=True` for CHW probabilities. Pass `mean` and `std` explicitly if
training used nondefault normalization. The CLI uses the configuration values.
Tensor `model(images)` expects already normalized inputs.

WSI inference reads tiled TIFFs lazily, stitches effective tile centers, handles
image boundaries, and writes a 256x256 tiled Zstd TIFF without allocating the
whole output in RAM. A sibling `mask.tif.work` directory retains chunked mask
and completion state after interruption. Calling the same command resumes only
when source identity, preprocessing, geometry and exact model weights match.
Existing final outputs are never overwritten. Successful publication removes
its reproducible scratch. Start with a new output path to change inputs.

## Import historical TiSAM weights

```python
from tisam import load_model
from tisam.checkpointing import export_weights

model = load_model("old-best.safetensors", device="cuda")
export_weights(model, "portable.safetensors")
```

Historical dual-encoder `.pt` and `.safetensors` weights are supported. If a file
has no architecture metadata, pass an explicit `ModelConfig`. Historical `.pt`
files containing arbitrary Python/NumPy state may require `trusted=True` (CLI:
`--trusted`); use that only for files you trust. Historical optimizer, scheduler,
benchmark manifests and run lineage are not imported.

Frozen encoder weights may be absent from these files. Loading reconstructs the
pretrained encoders first, then validates every required TiSAM parameter and
shape. Copying a decoder checkpoint alone does not make a deployment offline.

## Development

```sh
uv tool install prek
prek install
prek run --all-files
prek run --all-files --hook-stage pre-push
uv build
```

Hooks automatically prepare their dependencies from `uv.lock` and fail if the
lockfile needs updating. Commit-stage Ruff hooks install only the `dev` group;
type checks and offline tests run only at pre-push and include all optional
dependencies. CI disables hook synchronization after explicitly preparing its
environment to preserve the minimum-stack overrides.

GitHub Actions runs offline tests and type checks on Python 3.9, 3.12, and 3.14
for pull requests, pushes to `main`, and manual runs. Python 3.12 also runs lint,
format, lock and workflow checks, builds the distribution, and verifies the base
wheel's public imports and CLI in a clean environment without optional extras.
CI uses the locked dependencies, with explicit minimum-stack overrides on Python
3.9, and does not require GPU access or model weights. Training, checkpoint resume, inference, and loss tests run on every CI version.

Ordinary tests use small offline fixtures. Real pretrained-model checks require
cached SAM3, UNI2-h, UNI2-SEAL and Virchow2 weights and CUDA. Select an available
GPU (physical GPU 1 in this example) and run them explicitly:

```sh
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 uv run --no-sync pytest tests/test_real.py -m real -q
```

These checks cover a training step and multi-patch WSI inference for each
encoder. Set `TISAM_TEST_CHECKPOINT` to an existing TiSAM checkpoint to also check
GPU inference before and after exporting and reloading native weights. That
checkpoint check is skipped when the variable is unset. Tests do not download
weights. See [architecture and dependency boundaries](docs/architecture.md)
for package and checkpoint contracts.

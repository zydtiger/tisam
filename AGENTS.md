# Development rules

This repository contains only dual-encoder TiSAM, single-process training,
validation, and tile/WSI inference. Keep model imports independent of training
and image I/O extras. Preserve encoder/decoder parameter names and computation
when changing packaging. Do not add other model architectures, distributed
execution, client/server applications, research campaigns, datasets, or weights.

Use uv from the project root. This is one installable distribution with source
under src/tisam and tests under tests; it is not a uv workspace.
README.md owns public setup and commands; docs/architecture.md owns package and
checkpoint contracts. Keep all instructions self-contained. Examples must use
portable paths and generic class names. Never put credentials or model weights
in Git. Preserve required third-party license notices.

The source repository is public on GitHub at https://github.com/zydtiger/tisam.
Do not push, publish packages, create release tags, or upload artifacts without
explicit approval. A project license and package release policy have not been
selected. Repository publication does not authorize a package or tagged release.

The base branch is main. Initial project bootstrap may be prepared there.
Subsequent substantial changes use a sibling worktree and a category-prefixed
branch. Preserve unrelated changes. Commit subjects use one of feat, fix, docs,
refactor, test, build, conf followed by a colon and lowercase imperative summary.
No commits or merges are required merely to prepare an implementation.

Use module and key API docstrings, top-level imports, and torch.nn.functional
without an alias. Change only code needed by the task. Targeted test additions
are encouraged for behavioral or package-boundary changes. TIFF output uses
256x256 tiles and Zstd compression; preserve HxWxC image layout.

Mechanical checks live in .pre-commit-config.yaml. Install prek once with
`uv tool install prek`, then activate hooks with `prek install`. Run
`prek run --all-files` and `prek run --all-files --hook-stage pre-push` before
handoff. Real pretrained-model tests are opt-in and must not download weights
as part of ordinary tests. GitHub Actions runs the pre-commit stage on Python
3.12 and the pre-push stage on Python 3.12, 3.13 and 3.14 under Linux without
GPU or model downloads. Python 3.12 also builds the distribution and verifies
the base wheel in an isolated environment. Workflow validation belongs to the
pinned actionlint hook; do not duplicate hook commands in CI.

Dependabot checks Actions weekly and uv dependencies and hooks monthly. The
workflow's uv 0.10.x and prek 0.4.x inputs and Python matrix require manual
updates; Dependabot does not manage action inputs. No release automation is
configured.

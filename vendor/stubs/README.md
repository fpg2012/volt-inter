# torch / torchvision resolver stubs

**Machine-local workaround** — these are *not* real packages and contain no code.

The real CUDA-enabled torch 2.13.0 + torchvision 0.28.0 (cu130) used here are
hardlinked into `.venv` from `../gaussian-splatting/.venv` by
`scripts/seed_torch_from_gs.sh`, so `uv sync` must not download/install torch
from PyPI.

`pyproject.toml` therefore points `torch` / `torchvision` at these stubs
(`[tool.uv.sources]`), which only declare the correct name + version so the
dependency resolver is satisfied. See the comments in `pyproject.toml` and
`scripts/uv_sync.sh` for the full workflow.

To reproduce this setup on another machine you must either provide an
equivalent CUDA torch build, or remove these stub sources and restore the
upstream PyTorch indexes.

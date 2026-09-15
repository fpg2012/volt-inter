#!/usr/bin/env bash
# Seed torch from gaussian-splatting, then uv sync without replacing it.
#
# NOTE(local): UV_CACHE_DIR defaults to /workspace/uv_cache, a shared cache on
# this workstation. Override UV_CACHE_DIR to use a different location.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12
fi

bash "$ROOT/scripts/seed_torch_from_gs.sh"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/uv_cache}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"
# 12 threads / 32GB: keep nvcc from spawning every arch × all jobs.
export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export FORCE_CUDA="${FORCE_CUDA:-1}"

uv sync \
  --no-install-package torch \
  --no-install-package torchvision \
  --no-install-package flash-attn \
  "$@"

#!/usr/bin/env bash
# Hardlink torch/CUDA stack from gaussian-splatting/.venv into Volt/.venv.
# No network, almost no extra disk (hardlinks).
#
# NOTE(local): the default SRC / FA_SRC paths below are specific to this
# workstation (/workspace). Override with GS_VENV / FLASH_ATTN_ARCHIVE
# environment variables when running elsewhere.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${GS_VENV:-/workspace/lab/gaussian-splatting/.venv}/lib/python3.12/site-packages"
DST="$ROOT/.venv/lib/python3.12/site-packages"

if [[ ! -d "$SRC/torch" ]]; then
  echo "missing source torch: $SRC/torch" >&2
  exit 1
fi

mkdir -p "$DST"

names=(
  torch torch-2.13.0.dist-info torchgen functorch
  torchvision torchvision-0.28.0.dist-info torchvision.libs
  triton triton-3.7.1.dist-info
  nvidia
  nvidia_cublas-13.1.1.3.dist-info
  nvidia_cuda_cupti-13.0.85.dist-info
  nvidia_cuda_nvrtc-13.0.88.dist-info
  nvidia_cuda_runtime-13.0.96.dist-info
  nvidia_cudnn_cu13-9.20.0.48.dist-info
  nvidia_cufft-12.0.0.61.dist-info
  nvidia_cufile-1.15.1.6.dist-info
  nvidia_curand-10.4.0.35.dist-info
  nvidia_cusolver-12.0.4.66.dist-info
  nvidia_cusparse-12.6.3.3.dist-info
  nvidia_cusparselt_cu13-0.8.1.dist-info
  nvidia_nccl_cu13-2.29.7.dist-info
  nvidia_nvjitlink-13.3.33.dist-info
  nvidia_nvshmem_cu13-3.4.5.dist-info
  nvidia_nvtx-13.0.85.dist-info
  cuda
  cuda_bindings-13.3.1.dist-info
  cuda_pathfinder-1.5.6.dist-info
  cuda_toolkit-13.0.3.0.dist-info
)

echo "seeding torch stack"
echo "  from $SRC"
echo "  to   $DST"

for n in "${names[@]}"; do
  if [[ ! -e "$SRC/$n" ]]; then
    echo "skip missing $n" >&2
    continue
  fi
  rm -rf "$DST/$n"
  cp -a --link "$SRC/$n" "$DST/$n" 2>/dev/null || cp -a "$SRC/$n" "$DST/$n"
  echo "  linked $n"
done

# Prebuilt flash-attn for torch 2.13 + cu130 (already in uv cache).
# NOTE(local): uv cache archive path is machine-specific; override via FLASH_ATTN_ARCHIVE.
FA_SRC="${FLASH_ATTN_ARCHIVE:-/workspace/uv_cache/archive-v0/eI5qmFtx6_YFQS_g}"
if [[ -d "$FA_SRC/flash_attn" ]]; then
  rm -rf "$DST/flash_attn" "$DST/flash_attn-2.8.3.dist-info" \
    "$DST/flash_attn-2.8.3+cu130torch2.13.dist-info" "$DST/hopper" \
    "$DST"/flash_attn_2_cuda*.so
  for n in flash_attn flash_attn-2.8.3+cu130torch2.13.dist-info \
           flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so hopper; do
    [[ -e "$FA_SRC/$n" ]] || continue
    cp -a --link "$FA_SRC/$n" "$DST/$n" 2>/dev/null || cp -a "$FA_SRC/$n" "$DST/$n"
    echo "  linked $n"
  done
fi

"$ROOT/.venv/bin/python" - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
import flash_attn
print("flash_attn", flash_attn.__version__)
PY

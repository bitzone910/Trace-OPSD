#!/usr/bin/env bash
# Launch in the dedicated A800 environment with external inputs and outputs.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
env_name="trace-opsd"
hardware_profile="a800"
assets_root=""
work_root=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --assets-root) assets_root="${2:?--assets-root requires a value}"; shift 2 ;;
    --work-root) work_root="${2:?--work-root requires a value}"; shift 2 ;;
    --hardware-profile) hardware_profile="${2:?--hardware-profile requires a value}"; shift 2 ;;
    -h|--help)
      printf '%s\n' 'Usage: bash scripts/launch_training.sh [--assets-root DIR] [--work-root DIR] <command> [args...]'
      exit 0 ;;
    *) break ;;
  esac
done
[[ "$hardware_profile" == "a800" ]] || { printf 'Error: only A800 is supported.\n' >&2; exit 2; }
[[ $# -gt 0 ]] || { printf 'Error: a command is required.\n' >&2; exit 2; }
source "$script_dir/assets_env.sh"
trace_opsd_assets_init "$assets_root" "$work_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${TRACE_OPSD_HF_HOME:-${TRACE_OPSD_WORK_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${TRACE_OPSD_HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TRACE_OPSD_TORCH_HOME:-${TRACE_OPSD_WORK_ROOT}/cache/torch}"
export RAY_TMPDIR="${TRACE_OPSD_RAY_TMPDIR:-${TRACE_OPSD_WORK_ROOT}/runtime/ray}"
export TMPDIR="${TRACE_OPSD_TMPDIR:-${TRACE_OPSD_WORK_ROOT}/runtime/tmp}"
export WANDB_DIR="${TRACE_OPSD_WANDB_DIR:-${TRACE_OPSD_WORK_ROOT}/wandb}"
command -v conda >/dev/null 2>&1 || { printf 'Error: conda is required.\n' >&2; exit 1; }
env_python="$(conda run --no-capture-output -n "$env_name" python -c 'import sys; print(sys.executable)')" || {
  printf "Error: dedicated Conda environment '%s' is unavailable.\n" "$env_name" >&2; exit 1;
}
export TRACE_OPSD_CACHE_PYTHON="$env_python"
profile_value() { "$env_python" -m trace_opsd.hardware_profile --profile a800 get "$1"; }
expected_cache_arch="$(profile_value gpu.cache_arch)"
torch_cuda_arch_list="$(profile_value gpu.torch_cuda_arch_list)"
"$env_python" -m trace_opsd.hardware_profile --profile a800 validate-devices >/dev/null || {
  printf 'Error: visible GPUs must be A800.\n' >&2; exit 1;
}
source "$script_dir/gpu_cache_env.sh"
trace_opsd_configure_gpu_caches "$expected_cache_arch" "$torch_cuda_arch_list"
mkdir -p "$HF_DATASETS_CACHE" "$TORCH_HOME" "$RAY_TMPDIR" "$TMPDIR" "$WANDB_DIR"
printf '%s\n' "ASSETS=$TRACE_OPSD_ASSET_ROOT" "WORK=$TRACE_OPSD_WORK_ROOT" \
  "HF_HOME=$HF_HOME" "TRITON_CACHE_DIR=$TRITON_CACHE_DIR" "RAY_TMPDIR=$RAY_TMPDIR" \
  "TMPDIR=$TMPDIR" "WANDB_DIR=$WANDB_DIR" "TRACE_OPSD_GPU_ARCH=$TRACE_OPSD_GPU_ARCH"
exec conda run --no-capture-output -n "$env_name" "$@"

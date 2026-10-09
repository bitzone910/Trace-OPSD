#!/usr/bin/env bash
# Configure architecture-scoped CUDA/JIT caches for the active Trace-OPSD GPU.

set -euo pipefail

trace_opsd_configure_gpu_caches() {
  local cache_python="${TRACE_OPSD_CACHE_PYTHON:-python}"
  local work_root="${TRACE_OPSD_WORK_ROOT:?initialize external asset paths first}"
  local expected_capability="${1:-}"
  local requested_arch_list="${2:-}"
  local capability cache_root omp_threads

  capability="$("$cache_python" -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required to resolve the Trace-OPSD cache architecture")
major, minor = torch.cuda.get_device_capability(0)
print(f"sm{major}{minor}")
')"
  [[ "$capability" =~ ^sm[0-9]+$ ]] || {
    printf 'Error: invalid CUDA compute capability: %s\n' "$capability" >&2
    return 1
  }
  if [[ -n "$expected_capability" && "$capability" != "$expected_capability" ]]; then
    printf 'Error: selected hardware profile expects %s, found %s\n' \
      "$expected_capability" "$capability" >&2
    return 1
  fi

  cache_root="${TRACE_OPSD_GPU_CACHE_ROOT:-${work_root}/cache/gpu}/${capability}"
  export TRACE_OPSD_GPU_ARCH="$capability"
  export TRACE_OPSD_ACTIVE_GPU_CACHE_ROOT="$cache_root"
  export TRITON_CACHE_DIR="${cache_root}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
  export TORCH_EXTENSIONS_DIR="${cache_root}/torch-extensions"
  export CUDA_CACHE_PATH="${cache_root}/cuda"

  omp_threads="${TRACE_OPSD_OMP_NUM_THREADS:-${OMP_NUM_THREADS:-8}}"
  if [[ ! "$omp_threads" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Warning: replacing invalid OMP_NUM_THREADS=%q with 8.\n' \
      "$omp_threads" >&2
    omp_threads=8
  fi
  export OMP_NUM_THREADS="$omp_threads"

  case "$capability" in
    sm80) export TORCH_CUDA_ARCH_LIST="8.0" ;;
    *)
      printf 'Error: unsupported GPU architecture for Trace-OPSD: %s\n' \
        "$capability" >&2
      return 1
      ;;
  esac
  if [[ -n "$requested_arch_list" \
    && "$TORCH_CUDA_ARCH_LIST" != "$requested_arch_list" ]]; then
    printf 'Error: profile TORCH_CUDA_ARCH_LIST=%s does not match %s (%s)\n' \
      "$requested_arch_list" "$capability" "$TORCH_CUDA_ARCH_LIST" >&2
    return 1
  fi

  mkdir -p \
    "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"

  if [[ "$TRITON_CACHE_DIR" != "${cache_root}/triton" \
    || "$TORCHINDUCTOR_CACHE_DIR" != "${cache_root}/torchinductor" \
    || "$TORCH_EXTENSIONS_DIR" != "${cache_root}/torch-extensions" \
    || "$CUDA_CACHE_PATH" != "${cache_root}/cuda" ]]; then
    printf 'Error: GPU caches are not scoped to %s.\n' "$capability" >&2
    return 1
  fi
}

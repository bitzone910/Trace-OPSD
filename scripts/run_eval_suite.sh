#!/usr/bin/env bash
# Evaluate explicitly selected HF models in the dedicated A800 environment.
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' \
    'Usage: bash scripts/run_eval_suite.sh --models-file YAML --protocol greedy|sampling8|all --name NAME [options]' \
    '  --gpu ID                 Single A800 GPU (default: 0)' \
    '  --assets-root DIR        Inputs (default: ../assets or TRACE_OPSD_ASSET_ROOT)' \
    '  --work-root DIR          Outputs (default: assets-root/runs)' \
    '  --dataset-dir DIR        Fixed snapshot (default: data/evaluation)' \
    '  --output-root DIR        Explicit output directory instead of --name' \
    '  --preflight-only         Inspect without GPU services' \
    '  --limit-per-benchmark N  Diagnostic subset, excluded from formal comparison' \
    '  --samples-per-question N --temperature FLOAT --top-p FLOAT --top-k INT --seed INT --pass-ks CSV' \
    'Template: configs/evaluation/models.example.yaml (fill placeholders before running)'
  exit 0
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Resolve roots before launch_training.sh configures caches and environment.
assets_root=""; work_root=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --assets-root) assets_root="${args[i+1]:?--assets-root needs a directory}" ;;
    --work-root) work_root="${args[i+1]:?--work-root needs a directory}" ;;
  esac
done
source "$repo_root/scripts/assets_env.sh"
trace_opsd_assets_init "$assets_root" "$work_root"
exec bash "$repo_root/scripts/launch_training.sh" --hardware-profile a800 \
  python -m trace_opsd.evaluation.batch "$@"

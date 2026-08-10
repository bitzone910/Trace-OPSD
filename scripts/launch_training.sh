#!/usr/bin/env bash
# Launch a training command with ToolElite's canonical Hugging Face cache.

set -euo pipefail

data_root="${TOOLELITE_DATA_ROOT:-/root/autodl-tmp}"
export HF_HOME="${TOOLELITE_HF_HOME:-${data_root}/huggingface_cache}"
export HF_DATASETS_CACHE="${TOOLELITE_HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

mkdir -p "$HF_DATASETS_CACHE"

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/launch_training.sh <training command> [args...]" >&2
  echo "HF_HOME=$HF_HOME" >&2
  echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE" >&2
  exit 2
fi

echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
exec "$@"

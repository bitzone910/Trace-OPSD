#!/usr/bin/env bash
# Shared external asset paths. Sourcing this file does not create directories.
trace_opsd_assets_init() {
  local script_root assets_arg work_arg
  if compgen -e | grep -q '^TOOLELITE_'; then
    echo "Error: remove old TOOLELITE_ variables and use TRACE_OPSD_ variables." >&2
    return 1
  fi
  script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  assets_arg="${1:-${TRACE_OPSD_ASSET_ROOT:-$script_root/../assets}}"
  work_arg="${2:-${TRACE_OPSD_WORK_ROOT:-$assets_arg/runs}}"
  # realpath -m resolves paths even before their directories have been prepared.
  export TRACE_OPSD_ASSET_ROOT="$(realpath -m -- "$assets_arg")"
  export TRACE_OPSD_WORK_ROOT="$(realpath -m -- "$work_arg")"
  export TRACE_OPSD_GRPO_DATA_DIR="${TRACE_OPSD_GRPO_DATA_DIR:-$TRACE_OPSD_ASSET_ROOT/datasets/grpo}"
  export TRACE_OPSD_GENERAL_INDEX="${TRACE_OPSD_GENERAL_INDEX:-$TRACE_OPSD_ASSET_ROOT/retrieval/general_search/index/e5_Flat.index}"
  export TRACE_OPSD_GENERAL_CORPUS="${TRACE_OPSD_GENERAL_CORPUS:-$TRACE_OPSD_ASSET_ROOT/retrieval/general_search/corpus/search_r1_corpus.jsonl}"
  export TRACE_OPSD_RETRIEVER_MODEL="${TRACE_OPSD_RETRIEVER_MODEL:-$TRACE_OPSD_ASSET_ROOT/models/e5-base-v2}"
}

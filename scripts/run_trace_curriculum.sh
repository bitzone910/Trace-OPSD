#!/usr/bin/env bash
# Run one dynamically sampled TRACE stage, then exit after checkpoint validation.
# --max-num-gen-batches defaults to 10; an existing run only permits increases.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repo_root/scripts/run_grpo.sh" trace-train --curriculum "$@"

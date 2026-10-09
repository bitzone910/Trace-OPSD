#!/usr/bin/env bash
# Launch isolated evaluation services, then run a sealed math benchmark snapshot.
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ -n "${OMP_NUM_THREADS:-}" && ! "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Warning: replacing invalid OMP_NUM_THREADS=%s with 8.\n' "$OMP_NUM_THREADS" >&2
  export OMP_NUM_THREADS=8
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

model_path=""
run_dir=""
assets_root=""
work_root=""
dataset_dir="data/evaluation"
benchmarks="aime24,aime25,aime26,math500,gsm8k"
mode="tool"
thinking_mode="auto"
model_gpus="0"
retrieval_device="cpu"
limit=""
logs=""
concurrency="16"
samples_per_question=""
temperature=""
top_p=""
top_k=""
seed=""
pass_ks=""

usage() {
  printf '%s\n' \
    'Usage: bash scripts/run_eval.sh --model-path DIR --run-dir DIR [options]' \
    'Options: --dataset-dir DIR --benchmarks CSV --mode tool|direct' \
    '         --thinking-mode auto|on|off --model-gpus ID' \
    '         --assets-root DIR --work-root DIR --retrieval-device cpu --limit-per-benchmark N' \
    '         --logs-dir DIR --concurrency N' \
    '         --samples-per-question N (enable independent stochastic trajectories)' \
    '         --temperature FLOAT --top-p FLOAT --top-k INT --seed INT --pass-ks CSV'
}
die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) model_path="${2:?}"; shift 2 ;;
    --run-dir) run_dir="${2:?}"; shift 2 ;;
    --assets-root) assets_root="${2:?}"; shift 2 ;;
    --work-root) work_root="${2:?}"; shift 2 ;;
    --dataset-dir) dataset_dir="${2:?}"; shift 2 ;;
    --benchmarks) benchmarks="${2:?}"; shift 2 ;;
    --mode) mode="${2:?}"; shift 2 ;;
    --thinking-mode) thinking_mode="${2:?}"; shift 2 ;;
    --model-gpu|--model-gpus) model_gpus="${2:?}"; shift 2 ;;
    --retrieval-device) retrieval_device="${2:?}"; shift 2 ;;
    --limit-per-benchmark) limit="${2:?}"; shift 2 ;;
    --logs-dir) logs="${2:?}"; shift 2 ;;
    --concurrency) concurrency="${2:?}"; shift 2 ;;
    --samples-per-question) samples_per_question="${2:?}"; shift 2 ;;
    --temperature) temperature="${2:?}"; shift 2 ;;
    --top-p) top_p="${2:?}"; shift 2 ;;
    --top-k) top_k="${2:?}"; shift 2 ;;
    --seed) seed="${2:?}"; shift 2 ;;
    --pass-ks) pass_ks="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

source "$repo_root/scripts/assets_env.sh"
trace_opsd_assets_init "$assets_root" "$work_root"

[[ -n "$model_path" && -d "$model_path" ]] || die '--model-path must name a local model directory'
[[ -n "$run_dir" ]] || die '--run-dir is required'
run_dir="$(realpath -m -- "$run_dir")"
case "$run_dir/" in
  "$repo_root/"*) die '--run-dir must be outside the source repository' ;;
esac
[[ "$mode" == tool || "$mode" == direct ]] || die '--mode must be tool or direct'
[[ "$thinking_mode" == auto || "$thinking_mode" == on || "$thinking_mode" == off ]] || die 'invalid thinking mode'
[[ "$model_gpus" =~ ^[0-9]+$ ]] || die '--model-gpus must name one GPU ID (TP=1)'
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || die '--concurrency must be a positive integer'
[[ -z "$limit" || "$limit" =~ ^[1-9][0-9]*$ ]] || die '--limit-per-benchmark must be a positive integer'
[[ "$retrieval_device" == cpu ]] || die '--retrieval-device must be cpu for the A800 evaluation protocol'
if [[ -z "$samples_per_question" ]]; then
  [[ -z "$temperature$top_p$top_k$seed$pass_ks" ]] || \
    die 'sampling settings require --samples-per-question; greedy evaluation is unchanged'
else
  [[ "$samples_per_question" =~ ^[1-9][0-9]*$ ]] || die '--samples-per-question must be a positive integer'
  temperature="${temperature:-0.7}"
  top_p="${top_p:-0.95}"
  top_k="${top_k:--1}"
  seed="${seed:-42}"
  pass_ks="$(python - "$samples_per_question" "$temperature" "$top_p" "$top_k" "$seed" "$pass_ks" <<'PY_SAMPLING'
import math
import sys

try:
    n, temperature, top_p = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    top_k, seed = int(sys.argv[4]), int(sys.argv[5])
    ks = [int(value) for value in sys.argv[6].split(",")] if sys.argv[6] else [k for k in (1, 2, 4, 8) if k <= n]
    if n < 2:
        raise ValueError("--samples-per-question must be at least 2")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("--temperature must be finite and positive")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if top_k != -1 and top_k <= 0:
        raise ValueError("--top-k must be -1 or a positive integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("--seed must be an integer in [0, 2**32 - 1]")
    if not ks or len(set(ks)) != len(ks) or any(not 1 <= k <= n for k in ks):
        raise ValueError("--pass-ks must be distinct integers in [1, samples-per-question]")
    print(",".join(str(k) for k in sorted({*ks, n})))
except (ValueError, OverflowError) as exc:
    raise SystemExit("Error: invalid sampling settings: " + str(exc)) from exc
PY_SAMPLING
  )"
fi
tensor_parallel_size="$(awk -F, '{print NF}' <<<"$model_gpus")"
if [[ "$retrieval_device" != cpu && ",$model_gpus," == *",$retrieval_device,"* ]]; then
  die 'model and retrieval GPU must differ'
fi
command -v curl >/dev/null || die 'curl is required'
command -v python >/dev/null || die 'python is required in the active trace-opsd environment'
command -v setsid >/dev/null || die 'setsid (util-linux) is required for isolated service cleanup'

model_path="$(realpath "$model_path")"
index_path="$TRACE_OPSD_ASSET_ROOT/retrieval/general_search/index/e5_Flat.index"
corpus_path="$TRACE_OPSD_ASSET_ROOT/retrieval/general_search/corpus/search_r1_corpus.jsonl"
retriever_path="$TRACE_OPSD_ASSET_ROOT/models/e5-base-v2"
if [[ "$mode" == tool ]]; then
  [[ -f "$index_path" && -f "$corpus_path" && -d "$retriever_path" ]] || \
    die 'General Search index, corpus, or E5 model is missing'
fi

port_free() {
  python - "$1" <<'PY_PORT'
import socket
import sys

with socket.socket() as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY_PORT
}
port_free 8000 || die 'port 8000 is occupied; stop its service before evaluation'
if [[ "$mode" == tool ]]; then
  port_free 8002 || die 'port 8002 is occupied; stop its service before evaluation'
fi

[[ -n "$logs" ]] || logs="$TRACE_OPSD_WORK_ROOT/evaluation-logs/$(basename "$run_dir")"
mkdir -p "$logs"
logs="$(realpath "$logs")"
python - "$logs" <<'PY_ROTATE'
from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4

logs = Path(sys.argv[1])
names = ("vllm.log", "tools.log", "execution-metadata.json", "tool-sanity.json", "executor-setup.log")
previous = [logs / name for name in names if (logs / name).is_file()]
if previous:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex
    destination = logs / "history" / stamp
    destination.mkdir(parents=True)
    for source in previous:
        source.replace(destination / source.name)
PY_ROTATE
executor_args=()
if [[ "$mode" == tool ]]; then
  executor_uid="${EXECUTOR_UID:-}"
  executor_gid="${EXECUTOR_GID:-}"
  if [[ -z "$executor_uid" && -z "$executor_gid" && "$(id -u)" == 0 ]]; then
    executor_user="trace-opsd-exec"
    python_bin="$(python -c 'import sys; print(sys.executable)')"
    if id "$executor_user" >/dev/null 2>&1 && command -v runuser >/dev/null 2>&1 && \
      runuser -u "$executor_user" -- "$python_bin" -I -c 'import numpy' \
        >"$logs/executor-setup.log" 2>&1; then
      printf 'Reusing Python executor user: %s\n' "$executor_user"
    else
      bash "$repo_root/scripts/setup_executor_user.sh" "$executor_user" "$python_bin" \
        >"$logs/executor-setup.log" 2>&1 || {
          cat "$logs/executor-setup.log" >&2
          die 'executor setup failed; ensure acl is installed, then retry'
        }
    fi
    executor_uid="$(id -u "$executor_user")"
    executor_gid="$(id -g "$executor_user")"
  fi
  if [[ -n "$executor_uid" || -n "$executor_gid" ]]; then
    [[ "$executor_uid" =~ ^[0-9]+$ && "$executor_gid" =~ ^[0-9]+$ ]] || \
      die 'EXECUTOR_UID and EXECUTOR_GID must both be numeric'
    (( 10#$executor_uid > 0 && 10#$executor_gid > 0 )) || \
      die 'EXECUTOR_UID and EXECUTOR_GID must be non-root IDs'
    executor_args=(--executor_uid "$executor_uid" --executor_gid "$executor_gid")
  elif [[ "$(id -u)" == 0 ]]; then
    die 'tool service would run Python as root; configure EXECUTOR_UID and EXECUTOR_GID'
  fi
fi

model_pid=""
tool_pid=""
cleanup() {
  local pid running deadline=$((SECONDS + 20))
  local pids=("$model_pid" "$tool_pid")
  # Signal both sessions before waiting; the batch supervisor allows 30 seconds.
  for pid in "${pids[@]}"; do
    [[ -z "$pid" ]] || kill -TERM -- "-$pid" 2>/dev/null || true
  done
  while (( SECONDS < deadline )); do
    running=false
    for pid in "${pids[@]}"; do
      if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
        running=true
      fi
    done
    [[ "$running" == true ]] || break
    sleep 1
  done
  for pid in "${pids[@]}"; do
    [[ -z "$pid" ]] || kill -KILL -- "-$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
  done
  deadline=$((SECONDS + 5))
  while [[ -n "$model_pid" ]] && ! port_free 8000; do
    (( SECONDS < deadline )) || { printf 'Error: evaluation port 8000 was not released\n' >&2; return 1; }
    sleep 1
  done
  while [[ -n "$tool_pid" ]] && ! port_free 8002; do
    (( SECONDS < deadline )) || { printf 'Error: evaluation port 8002 was not released\n' >&2; return 1; }
    sleep 1
  done
}
finish() {
  local status=$?
  trap - EXIT
  cleanup || status=1
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid env CUDA_VISIBLE_DEVICES="$model_gpus" PYTHONUNBUFFERED=1 \
  python -m vllm.entrypoints.openai.api_server \
  --model "$model_path" --served-model-name trace-opsd-eval \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --tensor-parallel-size "$tensor_parallel_size" \
  --max-model-len 8192 --gpu-memory-utilization 0.8 \
  --generation-config vllm >"$logs/vllm.log" 2>&1 &
model_pid=$!

tool_args=()
if [[ "$mode" == tool ]]; then
  if [[ "$retrieval_device" == cpu ]]; then
    tool_cuda=""
    tool_args=(--device cpu --no-faiss_gpu)
  else
    tool_cuda="$retrieval_device"
    tool_args=(--device cuda --use_fp16 --faiss_gpu)
  fi
  setsid env CUDA_VISIBLE_DEVICES="$tool_cuda" PYTHONUNBUFFERED=1 \
    python -m trace_opsd.tools.service \
    --general_index_path "$index_path" --general_corpus_path "$corpus_path" \
    --retriever_model "$retriever_path" --search_topk 3 \
    "${tool_args[@]}" "${executor_args[@]}" \
    --host 127.0.0.1 --port 8002 >"$logs/tools.log" 2>&1 &
  tool_pid=$!
fi

wait_ready() {
  local url="$1" pid="$2" log="$3" deadline=$((SECONDS + 600))
  until curl -fsS --max-time 3 "$url" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null || (( SECONDS >= deadline )); then
      tail -n 60 "$log" >&2 || true
      die "service did not become healthy: $url"
    fi
    sleep 2
  done
}
wait_ready http://127.0.0.1:8000/health "$model_pid" "$logs/vllm.log"
if [[ "$mode" == tool ]]; then
  wait_ready http://127.0.0.1:8002/health "$tool_pid" "$logs/tools.log"
fi

# Attest the engine's actual initialization dtype, rather than config.json alone.
python - "$logs/vllm.log" "$logs/execution-metadata.json" "$tensor_parallel_size" "$concurrency" <<'PY_METADATA'
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import torch

log_path, output_path = map(Path, sys.argv[1:3])
evidence = None
for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    if re.search(r"Initializing\b.*\bengine\b", line, re.IGNORECASE):
        match = re.search(r"\bdtype\s*=\s*(torch\.)?(bfloat16|float16|float32)\b", line)
        if match:
            if match.group(2) != "bfloat16":
                raise SystemExit("Error: vLLM initialized with a non-BF16 dtype")
            quantization = re.search(r"\bquantization\s*=\s*([^,\s)]+)", line)
            if not quantization or quantization.group(1) != "None":
                raise SystemExit("Error: vLLM startup did not attest quantization=None")
            evidence = line
if evidence is None:
    raise SystemExit("Error: vLLM startup log does not attest actual BF16 dtype; inspect " + str(log_path))
metadata = {
    "schema": "trace_opsd.evaluation.execution.v1",
    "inference": {
        "dtype": "bfloat16", "tensor_parallel_size": int(sys.argv[3]),
        "quantization": None, "concurrency": int(sys.argv[4]),
    },
    "backend": {
        "vllm": importlib.metadata.version("vllm"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
    },
    "verification": {
        "method": "vllm_startup_log", "actual_dtype": "bfloat16", "evidence": evidence,
    },
}
temporary = output_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(output_path)
PY_METADATA

if [[ "$mode" == tool ]]; then
  python - "$logs/tool-sanity.json" <<'PY_SANITY'
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

payload = {"calls": [
    {"request_id": "python-sanity", "tool": "python", "raw_text": "<python>print(2 + 2)</python>"},
    {"request_id": "search-sanity", "tool": "search", "raw_text": "<search>Pythagorean theorem</search>"},
]}
request = Request("http://127.0.0.1:8002/tools/batch", data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"}, method="POST")
with urlopen(request, timeout=120) as response:
    result = json.load(response)
responses = {item.get("request_id"): item for item in result.get("results", [])}
python_result = responses.get("python-sanity", {})
search_result = responses.get("search-sanity", {})
if python_result.get("status") != "ok" or python_result.get("result", "").strip() != "<result>\n4\n</result>":
    raise SystemExit("Error: Python execution sanity check failed: " + json.dumps(python_result))
if search_result.get("status") != "ok" or not search_result.get("result", "").strip():
    raise SystemExit("Error: General Search sanity check failed: " + json.dumps(search_result))
Path(sys.argv[1]).write_text(json.dumps({"status": "passed", "results": result["results"]},
                                      ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY_SANITY
fi

eval_args=(--model-path "$model_path" --run-dir "$run_dir" \
  --dataset-dir "$dataset_dir" --benchmarks "$benchmarks" \
  --index-path "$index_path" --corpus-path "$corpus_path" \
  --mode "$mode" --thinking-mode "$thinking_mode" --concurrency "$concurrency" \
  --execution-metadata "$logs/execution-metadata.json")
[[ -z "$limit" ]] || eval_args+=(--limit-per-benchmark "$limit")
if [[ -n "$samples_per_question" ]]; then
  eval_args+=(--samples-per-question "$samples_per_question" \
    --temperature "$temperature" --top-p "$top_p" --top-k "$top_k" \
    --seed "$seed" --pass-ks "$pass_ks")
  python -m trace_opsd.evaluation sample "${eval_args[@]}"
else
  python -m trace_opsd.evaluation run "${eval_args[@]}"
fi

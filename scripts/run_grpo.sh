#!/usr/bin/env bash
# Validate, train stages, resume, and export TRACE-OPSD on A800.

set -euo pipefail
original_args=("$@")

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
default_hardware_profile="a800"
hardware_profile="$default_hardware_profile"
runtime_config=""
assets_root=""
work_root=""
resume="auto"
curriculum_after=""
curriculum_mid=""
curriculum_high=""
curriculum_source=""
curriculum_requested=false
max_num_gen_batches=10
dynamic_budget_requested=false
tool_base_url="${TRACE_OPSD_TOOL_BASE_URL:-http://127.0.0.1:8002}"
tool_base_url="${tool_base_url%/}"
tool_host="127.0.0.1"
if [[ "$tool_base_url" =~ ^http://127\.0\.0\.1:([0-9]+)$ ]]; then
  tool_port="${BASH_REMATCH[1]}"
else
  printf 'Error: TRACE_OPSD_TOOL_BASE_URL must be local http://127.0.0.1:PORT\n' >&2
  exit 1
fi
(( 10#$tool_port >= 1 && 10#$tool_port <= 65535 )) || {
  printf 'Error: invalid tool service port: %s\n' "$tool_port" >&2
  exit 1
}

export PYTHONPATH="$repo_root/frameworks/verl:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export TRACE_OPSD_TOOL_BASE_URL="$tool_base_url"
export WANDB_MODE="${WANDB_MODE:-offline}"
# Use the same explicit BF16 attention backend for all A800 stages.
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/run_trace_curriculum.sh --after-step STEP --mid-per-step N --high-per-step N [options] [-- HYDRA...]
  bash scripts/run_grpo.sh validate [--model-path MODEL] [--dataset-dir DATA]
  bash scripts/run_grpo.sh export --run-name NAME [--checkpoint latest|best|STEP] [--target-dir DIR]

Options: --assets-root DIR --work-root DIR --run-name NAME --model-path DIR
         --dataset-dir DIR --runtime-config FILE --resume auto|none|latest
         --difficulty-source PARQUET --max-num-gen-batches N --thinking-mode auto|on|off
Default: 4 A800 GPUs, CPU retrieval, SFT-Qwen2.5-3B under ASSETS/models.
TRACE runs 50 steps per stage (ending at global step 622), then validates and exits.
Set mid + high = 32 explicitly for each stage. Mixed groups are retained with
up to 10 dynamic generation batches per actor update. Old token-credit runs
are read-only: export their HF weights to initialize a new turn-mean run.

EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

[[ $# -ge 1 ]] || { usage; exit 2; }
command_name="$1"
shift

case "$command_name" in
  -h|--help) usage; exit 0 ;;
  validate|trace-train|export) ;;
  *) die "supported commands are validate, trace-train and export" ;;
esac

filtered_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --assets-root) assets_root="${2:?--assets-root requires a value}"; shift 2 ;;
    --work-root) work_root="${2:?--work-root requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --hardware-profile)
      [[ $# -ge 2 ]] || die "--hardware-profile requires a value"
      hardware_profile="$2"
      shift 2
      ;;
    --runtime-config)
      [[ $# -ge 2 ]] || die "--runtime-config requires a value"
      runtime_config="$2"
      shift 2
      ;;
    --resume)
      [[ $# -ge 2 ]] || die "--resume requires a value"
      resume="$2"
      shift 2
      ;;
    --after-step)
      [[ $# -ge 2 ]] || die "--after-step requires a value"
      curriculum_after="$2"
      shift 2
      ;;
    --mid-per-step)
      [[ $# -ge 2 ]] || die "--mid-per-step requires a value"
      curriculum_mid="$2"
      shift 2
      ;;
    --high-per-step)
      [[ $# -ge 2 ]] || die "--high-per-step requires a value"
      curriculum_high="$2"
      shift 2
      ;;
    --difficulty-source)
      [[ $# -ge 2 ]] || die "--difficulty-source requires a value"
      curriculum_source="$2"
      shift 2
      ;;
    --curriculum)
      curriculum_requested=true
      shift
      ;;
    --max-num-gen-batches)
      [[ $# -ge 2 ]] || die "--max-num-gen-batches requires a value"
      max_num_gen_batches="$2"
      dynamic_budget_requested=true
      [[ "$max_num_gen_batches" =~ ^[1-9][0-9]*$ ]] || die "--max-num-gen-batches must be a positive integer"
      shift 2
      ;;
    *) filtered_args+=("$1"); shift ;;
  esac
done
[[ "$hardware_profile" == "a800" ]] || die "only A800 is supported"
source "$repo_root/scripts/assets_env.sh"
trace_opsd_assets_init "$assets_root" "$work_root"
if [[ "$dynamic_budget_requested" == true && "$command_name" != "trace-train" ]]; then
  die "--max-num-gen-batches requires TRACE training"
fi
if [[ "$command_name" == "trace-train" || "$curriculum_requested" == true || -n "$curriculum_after$curriculum_mid$curriculum_high$curriculum_source" ]]; then
  curriculum_requested=true
  [[ "$command_name" == "trace-train" ]] || die "curriculum options require TRACE training"
  [[ -n "$curriculum_after" && -n "$curriculum_mid" && -n "$curriculum_high" ]] || die \
    "curriculum requires --after-step, --mid-per-step, and --high-per-step"
  [[ "$curriculum_after" =~ ^[0-9]+$ && "$curriculum_mid" =~ ^[0-9]+$ && "$curriculum_high" =~ ^[0-9]+$ ]] || die \
    "curriculum step and difficulty counts must be non-negative integers"
  (( 10#$curriculum_mid + 10#$curriculum_high == 32 )) || die \
    "formal TRACE requires mid-per-step + high-per-step = 32"
  (( 10#$curriculum_after < 622 && 10#$curriculum_after % 50 == 0 )) || die \
    "after-step must be a stage boundary: 0, 50, ..., 600"
  if [[ -z "$curriculum_source" ]]; then
    curriculum_source="$TRACE_OPSD_ASSET_ROOT/datasets/numinamath/rl_trace_opsd.parquet"
  fi
  if [[ "$curriculum_after" == "0" ]]; then
    [[ "$resume" == "auto" || "$resume" == "none" || "$resume" == "latest" ]] || die "first stage requires --resume auto, none, or latest"
  else
    [[ "$resume" == "auto" || "$resume" == "latest" ]] || die "curriculum resume requires --resume latest"
    resume="latest"
  fi
fi
if [[ -z "$runtime_config" ]]; then
  runtime_config="$(
    conda run --no-capture-output -n trace-opsd \
      python -m trace_opsd.hardware_profile --profile "$hardware_profile" \
      get runtime_config
  )" || die "invalid hardware profile: $hardware_profile"
fi
runtime_config="$(cd "$(dirname "$runtime_config")" && pwd)/$(basename "$runtime_config")"
profile_cache_arch="$(
  conda run --no-capture-output -n trace-opsd \
    python -m trace_opsd.hardware_profile --profile "$hardware_profile" \
    get gpu.cache_arch
)"
profile_arch_list="$(
  conda run --no-capture-output -n trace-opsd \
    python -m trace_opsd.hardware_profile --profile "$hardware_profile" \
    get gpu.torch_cuda_arch_list
)"
profile_attention_backend="$(
  conda run --no-capture-output -n trace-opsd \
    python -m trace_opsd.hardware_profile --profile "$hardware_profile" \
    get inference.vllm_attention_backend
)"
[[ "$profile_attention_backend" == "FLASH_ATTN" ]] || die \
  "hardware profile must require the FLASH_ATTN vLLM backend"
cache_python="$(
  conda run --no-capture-output -n trace-opsd python -c \
    'import sys; print(sys.executable)'
)"
export TRACE_OPSD_CACHE_PYTHON="$cache_python"
source "$repo_root/scripts/gpu_cache_env.sh"
trace_opsd_configure_gpu_caches "$profile_cache_arch" "$profile_arch_list"

runtime_value() {
  conda run --no-capture-output -n trace-opsd \
    python -m trace_opsd.runtime_config --config "$runtime_config" get "$1"
}

gpu_count="$(runtime_value grpo.gpu_count)"
retrieval_mode="$(runtime_value grpo.retrieval_mode)"
rollout_tp="$(runtime_value grpo.rollout_tensor_parallel_size)"
checkpoint_root="$(runtime_value checkpoint.grpo_root)"
export_root="$(runtime_value checkpoint.export_root)"
keep_last="$(runtime_value checkpoint.grpo_keep_last)"
[[ "$gpu_count" =~ ^[1-8]$ ]] || die "invalid grpo.gpu_count: $gpu_count"
[[ "$rollout_tp" == "1" ]] || die "GRPO rollout tensor parallel size must be 1"

if [[ "$command_name" == "export" ]]; then
  exec bash "$repo_root/scripts/launch_training.sh" \
    --hardware-profile "$hardware_profile" \
    python -m trace_opsd.grpo.cli export \
    --runtime-config "$runtime_config" \
    --checkpoint-root "$checkpoint_root" "${filtered_args[@]}"
fi

case "$command_name" in
  validate|trace-train) ;;
  *) usage; exit 2 ;;
esac

algorithm_name="trace_opsd"
profile_name="train"
trainer_module="trace_opsd.grpo.trace_main"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a available_devices <<<"$CUDA_VISIBLE_DEVICES"
else
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
  mapfile -t available_devices < <(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' '
  )
fi
if [[ "$retrieval_mode" == "local_dedicated" ]]; then
  required_gpus=$((gpu_count + 1))
else
  [[ "$retrieval_mode" == "local_cpu" ]] || die "invalid retrieval mode: $retrieval_mode"
  required_gpus="$gpu_count"
fi
(( ${#available_devices[@]} >= required_gpus )) || die \
  "grpo.gpu_count=$gpu_count with retrieval_mode=$retrieval_mode requires $required_gpus visible GPUs; found ${#available_devices[@]}"
training_devices="$(IFS=,; printf '%s' "${available_devices[*]:0:gpu_count}")"
retrieval_device=""
retrieval_uuid="cpu"
selected_devices="$training_devices"
if [[ "$retrieval_mode" == "local_dedicated" ]]; then
  retrieval_device="${available_devices[$gpu_count]}"
  selected_devices="${training_devices},${retrieval_device}"
fi
env CUDA_VISIBLE_DEVICES="$selected_devices" \
  conda run --no-capture-output -n trace-opsd \
  python -m trace_opsd.hardware_profile --profile "$hardware_profile" \
  validate-devices --expected-count "$required_gpus" >/dev/null || die \
  "selected GRPO GPUs do not match $hardware_profile"
if [[ "$retrieval_mode" == "local_dedicated" ]]; then
  if [[ "$retrieval_device" == GPU-* || "$retrieval_device" == MIG-* ]]; then
    retrieval_uuid="$retrieval_device"
  else
    retrieval_uuid="$(
      nvidia-smi --id="$retrieval_device" --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d ' '
    )"
  fi
  [[ -n "$retrieval_uuid" ]] || die "could not resolve retrieval GPU UUID"
fi

mkdir -p "$checkpoint_root" "$export_root"
printf 'Trace-OPSD project: %s\n' "$repo_root"
printf 'GRPO checkpoint root: %s\n' "$checkpoint_root"
printf 'Training GPUs (%s): %s\n' "$gpu_count" "$training_devices"
if [[ "$retrieval_mode" == "local_cpu" ]]; then
  printf 'Retrieval: CPU (E5 encoder + FAISS index)\n'
else
  printf 'Retrieval GPU: %s (%s)\n' "$retrieval_device" "$retrieval_uuid"
fi
printf 'Hardware profile: %s\n' "$hardware_profile"
df -h "$checkpoint_root" | tail -n 1

model_path="${TRACE_OPSD_GRPO_MODEL:-$TRACE_OPSD_ASSET_ROOT/models/SFT-Qwen2.5-3B}"
dataset_dir="${TRACE_OPSD_GRPO_DATA_DIR:-$TRACE_OPSD_ASSET_ROOT/datasets/grpo}"
thinking_mode="auto"
run_name="trace-opsd"
hydra_overrides=()
set -- "${filtered_args[@]}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) [[ $# -ge 2 ]] || die "--model-path requires a value"; model_path="$2"; shift 2 ;;
    --dataset-dir) [[ $# -ge 2 ]] || die "--dataset-dir requires a value"; dataset_dir="$2"; shift 2 ;;
    --thinking-mode) [[ $# -ge 2 ]] || die "--thinking-mode requires a value"; thinking_mode="$2"; shift 2 ;;
    --run-name) [[ $# -ge 2 ]] || die "--run-name requires a value"; run_name="$2"; shift 2 ;;
    --) shift; hydra_overrides=("$@"); break ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$model_path" ]] || die "--model-path or TRACE_OPSD_GRPO_MODEL is required"
[[ "$thinking_mode" == "auto" || "$thinking_mode" == "on" || "$thinking_mode" == "off" ]] || \
  die "--thinking-mode must be auto, on, or off"

for override in "${hydra_overrides[@]}"; do
  normalized_override="${override#+}"
  normalized_override="${normalized_override#+}"
  normalized_override="${normalized_override#~}"
  if [[ "$normalized_override" == distillation.* || \
        "$normalized_override" == distillation=* ]]; then
    if [[ "$normalized_override" == "distillation.enabled=false" || \
          "$normalized_override" == "distillation.enabled=False" ]]; then
      continue
    fi
    die "external distillation teacher overrides are not supported: $override"
  fi
done

preflight=(
  python -m trace_opsd.grpo.cli validate
  --model-path "$model_path"
  --dataset-dir "$dataset_dir"
  --algorithm "$algorithm_name"
  --thinking-mode "$thinking_mode"
)
if [[ "$command_name" == "validate" ]]; then
  exec env CUDA_VISIBLE_DEVICES="$training_devices" \
    bash "$repo_root/scripts/launch_training.sh" \
    --hardware-profile "$hardware_profile" "${preflight[@]}"
fi
[[ -n "$run_name" ]] || die "--run-name is required for $command_name"

env CUDA_VISIBLE_DEVICES="$training_devices" \
  bash "$repo_root/scripts/launch_training.sh" \
  --hardware-profile "$hardware_profile" \
  python -m trace_opsd.grpo.cli record-run \
  --profile "$profile_name" \
  --algorithm "$algorithm_name" \
  --run-name "$run_name" \
  --checkpoint-root "$checkpoint_root" \
  --runtime-config "$runtime_config" \
  --training-gpus "$training_devices" \
  --retrieval-gpu-uuid "$retrieval_uuid" \
  --retrieval-mode "$retrieval_mode" \
  --model-path "$model_path" \
  --dataset-dir "$dataset_dir" \
  --thinking-mode "$thinking_mode" \
  --max-num-gen-batches "$max_num_gen_batches"

run_dir="$checkpoint_root/$run_name"
revalidate_previous=false
package_trace_stage() {
  # No GPU/model reload: prefer direct metrics, then console/W&B offline history.
  "$cache_python" "$repo_root/scripts/package_trace_stage.py" \
    --project-root "$repo_root" \
    --work-root "$TRACE_OPSD_WORK_ROOT" \
    --run-name "$run_name" --run-dir "$run_dir" \
    --after-step "$curriculum_after" --step "$curriculum_end" \
    --log-file "$TRACE_OPSD_WORK_ROOT/logs/$run_name-stage-$curriculum_after.log" \
    --log-file "$run_dir/console-stage-$curriculum_after.log"
}
if [[ "$curriculum_requested" == true ]]; then
  curriculum_export_args=()
  if [[ "$profile_name" == "train" ]]; then
    curriculum_export_args=(--export-dir "$export_root/$run_name")
  fi
  curriculum_result="$(
    "$cache_python" -m trace_opsd.grpo.curriculum prepare \
      --run-dir "$run_dir" --dataset-dir "$dataset_dir" \
      --difficulty-source "$curriculum_source" \
      --after-step "$curriculum_after" \
      --mid-per-step "$curriculum_mid" \
      --high-per-step "$curriculum_high" "${curriculum_export_args[@]}"
  )" || die "TRACE curriculum preparation failed"
  curriculum_action="$("$cache_python" -c 'import json,sys; print(json.loads(sys.argv[1])["action"])' "$curriculum_result")"
  curriculum_end="$("$cache_python" -c 'import json,sys; print(json.loads(sys.argv[1])["end_step"])' "$curriculum_result")"
  if [[ "$curriculum_action" == "revalidate_previous" ]]; then
    revalidate_previous=true
    curriculum_after="$("$cache_python" -c 'import json,sys; print(json.loads(sys.argv[1])["start_step"])' "$curriculum_result")"
    curriculum_mid="$("$cache_python" -c 'import json,sys; print(json.loads(sys.argv[1])["mid_per_step"])' "$curriculum_result")"
    curriculum_high="$("$cache_python" -c 'import json,sys; print(json.loads(sys.argv[1])["high_per_step"])' "$curriculum_result")"
    curriculum_action="revalidate"
  fi
  printf 'TRACE curriculum: %s, steps %s-%s, mid=%s, high=%s\n' \
    "$curriculum_action" "$curriculum_after" "$curriculum_end" "$curriculum_mid" "$curriculum_high"
  if [[ "$curriculum_action" == "complete" ]]; then
    if [[ "$profile_name" == "train" ]]; then
      package_trace_stage
    fi
    exit 0
  fi
  curriculum_seed="$(
    "$cache_python" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["sampling"]["seed"])' \
      "$run_dir/trace_curriculum.json"
  )"
  if [[ "$resume" == "none" && -f "$run_dir/latest_checkpointed_iteration.txt" ]]; then
    die "existing curriculum checkpoint requires --resume auto or latest"
  fi
  if [[ "$resume" == "auto" && ! -f "$run_dir/latest_checkpointed_iteration.txt" ]]; then
    resume="none"
  fi
fi
run_manifest="$run_dir/trace_opsd_grpo_run.json"
model_type="$(
  "$cache_python" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["model_type"])' \
    "$run_manifest"
)"
resolved_thinking_mode="$(
  "$cache_python" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["thinking_mode"])' \
    "$run_manifest"
)"
case "$resume" in
  none) resume_mode="disable"; resume_path="" ;;
  auto) resume_mode="auto"; resume_path="" ;;
  latest)
    [[ -f "$run_dir/latest_checkpointed_iteration.txt" ]] || die \
      "--resume latest requested but no committed checkpoint exists in $run_dir"
    resume_mode="auto"
    resume_path=""
    ;;
  *) die "TRACE resume must be auto, none, or latest" ;;

esac

started_tool_pid=""
tool_log_dir="$TRACE_OPSD_WORK_ROOT/runtime/grpo/$run_name"
mkdir -p "$tool_log_dir"
tool_log="$tool_log_dir/tool-service.log"

cleanup() {
  local exit_code="$?"
  trap - EXIT INT TERM
  if [[ -n "$started_tool_pid" ]] && kill -0 "$started_tool_pid" 2>/dev/null; then
    kill -TERM "$started_tool_pid" 2>/dev/null || true
    wait "$started_tool_pid" 2>/dev/null || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

tool_health_payload() {
  curl --fail --silent --max-time 3 "$tool_base_url/health"
}

tool_ready() {
  local payload
  payload="$(tool_health_payload)" || return 1
  EXPECTED_GPU_UUID="$retrieval_uuid" EXPECTED_RETRIEVAL_MODE="$retrieval_mode" python -c '
import json, os, sys
p=json.load(sys.stdin); gpu=p.get("gpu", {})
searches=(p.get("search", {}),)
common=(p.get("status")=="ok" and p.get("service_version")=="2.0.0"
        and p.get("batch_endpoint")=="/tools/batch")
if os.environ["EXPECTED_RETRIEVAL_MODE"] == "local_cpu":
    placement=(all(x.get("device")=="cpu" for x in searches)
               and all(x.get("faiss_device")=="cpu" and x.get("faiss_gpus")==0 for x in searches))
else:
    placement=(gpu.get("physical_gpu_uuid")==os.environ["EXPECTED_GPU_UUID"]
               and gpu.get("device_count")==1
               and all(x.get("device", "").startswith("cuda") for x in searches)
               and all(x.get("faiss_device")=="gpu" and x.get("faiss_gpus")==1 for x in searches))
ok=common and placement
raise SystemExit(0 if ok else 1)
' <<<"$payload" >/dev/null 2>&1
}

resolve_repo_path() {
  if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$repo_root" "$1"; fi
}

if tool_ready; then
  printf 'Reusing verified Trace-OPSD tool service (%s).\n' "$retrieval_mode"
elif tool_health_payload >/dev/null 2>&1; then
  die "a tool service is already listening at $tool_base_url but its GPU identity/configuration does not match"
else
  general_index="$(resolve_repo_path "${TRACE_OPSD_GENERAL_INDEX}")"
  general_corpus="$(resolve_repo_path "${TRACE_OPSD_GENERAL_CORPUS}")"
  retriever_model="$(resolve_repo_path "${TRACE_OPSD_RETRIEVER_MODEL}")"
  [[ -f "$general_index" ]] || die "general index not found: $general_index"
  [[ -f "$general_corpus" ]] || die "general corpus not found: $general_corpus"
  [[ -d "$retriever_model" ]] || die "retriever model not found: $retriever_model"

  executor_args=()
  executor_uid="${EXECUTOR_UID:-}"
  executor_gid="${EXECUTOR_GID:-}"
  if [[ -z "$executor_uid" && -z "$executor_gid" && "$(id -u)" == "0" ]]; then
    executor_user="trace-opsd-exec"
    bash "$repo_root/scripts/setup_executor_user.sh" "$executor_user" "$cache_python" || die \
      "executor setup failed; install acl if setfacl is missing, then retry"
    executor_uid="$(id -u "$executor_user")"
    executor_gid="$(id -g "$executor_user")"
  fi
  if [[ -n "$executor_uid" || -n "$executor_gid" ]]; then
    [[ -n "$executor_uid" && -n "$executor_gid" ]] || die \
      "EXECUTOR_UID and EXECUTOR_GID must be configured together"
    executor_args=(--executor_uid "$executor_uid" --executor_gid "$executor_gid")
  elif [[ "$(id -u)" == "0" ]]; then
    die "tool service would run Python as root; configure EXECUTOR_UID and EXECUTOR_GID"
  fi

  tool_device="cuda"
  tool_precision_args=(--use_fp16 --faiss_gpu)
  tool_cuda_devices="$retrieval_device"
  tool_physical_uuid="$retrieval_uuid"
  if [[ "$retrieval_mode" == "local_cpu" ]]; then
    tool_device="cpu"
    tool_precision_args=(--no-faiss_gpu)
    tool_cuda_devices=""
    tool_physical_uuid=""
  fi

  (
    cd "$repo_root"
    exec env CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES="$tool_cuda_devices" \
      TRACE_OPSD_PHYSICAL_GPU_UUID="$tool_physical_uuid" \
      PYTHONUNBUFFERED=1 \
      conda run --no-capture-output -n trace-opsd \
      python -m trace_opsd.tools.service \
      --general_index_path "$general_index" \
      --general_corpus_path "$general_corpus" \
      --retriever_model "$retriever_model" \
      --device "$tool_device" --max_length 256 --retrieval_batch_size 128 \
      --corpus_num_proc 4 --search_topk 3 \
      "${tool_precision_args[@]}" --result_max_chars 100000 \
      --raw_text_max_chars 100000 --python_code_max_chars 20000 \
      --search_query_max_chars 2000 --python_timeout 10 \
      --python_cpu_seconds 10 --python_memory_mb 512 \
      --python_output_bytes 65536 --python_nofile 64 \
      --python_max_concurrency 8 --python_queue_timeout 30 \
      --host "$tool_host" --port "$tool_port" "${executor_args[@]}"
  ) >>"$tool_log" 2>&1 &
  started_tool_pid="$!"
  deadline=$((SECONDS + 600))
  until tool_ready; do
    if ! kill -0 "$started_tool_pid" 2>/dev/null; then
      tail -n 80 "$tool_log" >&2 || true
      die "tool service exited before becoming healthy"
    fi
    (( SECONDS < deadline )) || { tail -n 80 "$tool_log" >&2 || true; die "tool service health timeout"; }
    sleep 2
  done
  printf 'Started Trace-OPSD tool service (PID %s); log: %s\n' "$started_tool_pid" "$tool_log"
fi

train_batch=32
val_batch=64
rollout_n=8
micro_batch=1
total_steps=622
total_epochs=1
rollout_gpu_memory_utilization=0.30
save_freq=50
test_freq=50

actor_checkpoint_keep="$keep_last"
val_before_train=true
if [[ "$algorithm_name" == "trace_opsd" && "$profile_name" == "train" ]]; then
  actor_checkpoint_keep=null
  # Establish a step-0 baseline only for a genuinely fresh TRACE run. A
  # resumed run starts from an already-trained checkpoint, so validating here
  # would incorrectly relabel that checkpoint as the initial-model baseline.
  if [[ "$resume_mode" == "disable" || \
        ( "$resume_mode" == "auto" && ! -f "$run_dir/latest_checkpointed_iteration.txt" ) ]]; then
    val_before_train=true
  else
    val_before_train=false
  fi
fi

greatest_common_divisor() {
  local left="$1" right="$2" remainder
  while (( right != 0 )); do
    remainder=$((left % right))
    left="$right"
    right="$remainder"
  done
  printf '%s\n' "$left"
}

# The trainer adds only the minimum prompt padding needed after accounting for
# rollout_n. Use that effective full prompt batch as a single PPO mini-batch so
# no trailing partial mini-batch is introduced for 1-7 training GPUs.
rollout_gpu_gcd="$(greatest_common_divisor "$gpu_count" "$rollout_n")"
prompt_divisor=$((gpu_count / rollout_gpu_gcd))
effective_train_batch=$(((train_batch + prompt_divisor - 1) / prompt_divisor * prompt_divisor))
mini_batch="$effective_train_batch"
val_batch=$(((val_batch + gpu_count - 1) / gpu_count * gpu_count))
printf 'GRPO batches: prompts=%s, padded_prompts=%s, rollout_n=%s, validation=%s\n' \
  "$train_batch" "$effective_train_batch" "$rollout_n" "$val_batch"

verl_args=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=false
  data.train_files="$dataset_dir/train.parquet"
  data.val_files="$dataset_dir/validation.parquet"
  data.train_batch_size="$train_batch"
  data.val_batch_size="$val_batch"
  data.seed=42
  data.max_prompt_length=2048
  data.max_response_length=4096
  data.filter_overlong_prompts=false
  data.truncation=error
  actor_rollout_ref.model.path="$model_path"
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.optim.lr=5e-7
  actor_rollout_ref.actor.optim.lr_warmup_steps=50
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
  actor_rollout_ref.actor.optim.min_lr_ratio=0.1
  actor_rollout_ref.actor.optim.num_cycles=0.5
  actor_rollout_ref.actor.ppo_mini_batch_size="$mini_batch"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$micro_batch"
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.use_kl_loss=true
  actor_rollout_ref.actor.kl_loss_coef=0.005
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.use_dynamic_bsz=true
  actor_rollout_ref.actor.fsdp_config.param_offload=false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]'
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.dtype=bfloat16
  actor_rollout_ref.rollout.tensor_model_parallel_size="$rollout_tp"
  actor_rollout_ref.rollout.gpu_memory_utilization="$rollout_gpu_memory_utilization"
  actor_rollout_ref.rollout.max_model_len=8192
  actor_rollout_ref.rollout.n="$rollout_n"
  actor_rollout_ref.rollout.temperature=1
  actor_rollout_ref.rollout.top_p=1
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.ignore_eos=false
  actor_rollout_ref.rollout.do_sample=true
  actor_rollout_ref.rollout.enable_prefix_caching=true
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$micro_batch"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true
  actor_rollout_ref.rollout.free_cache_engine=true
  actor_rollout_ref.rollout.agent.default_agent_loop=trace_opsd_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$repo_root/configs/grpo/agent_loop_trace_train.yaml"
  actor_rollout_ref.rollout.agent.num_workers=8
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$micro_batch"
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  reward.num_workers=8
  reward.reward_model.enable=false
  reward.reward_manager.source=importlib
  reward.reward_manager.name=TraceOPSDSearchR1EMRewardManager
  reward.reward_manager.module.path="$repo_root/trace_opsd/grpo/reward.py"
  distillation.enabled=false
  'trainer.logger=[console,wandb]'
  trainer.project_name=Trace-OPSD
  trainer.experiment_name="$run_name"
  trainer.n_gpus_per_node="$gpu_count"
  trainer.nnodes=1
  trainer.total_epochs="$total_epochs"
  trainer.total_training_steps="$total_steps"
  trainer.save_freq="$save_freq"
  trainer.test_freq="$test_freq"
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.temperature=0
  actor_rollout_ref.rollout.val_kwargs.do_sample=false
  trainer.val_before_train="$val_before_train"
  trainer.resume_mode="$resume_mode"
  trainer.default_local_dir="$run_dir"
  trainer.max_actor_ckpt_to_keep="$actor_checkpoint_keep"
  trainer.rollout_data_dir="$TRACE_OPSD_WORK_ROOT/grpo/$run_name/rollouts"
  trainer.validation_data_dir="$TRACE_OPSD_WORK_ROOT/grpo/$run_name/validation"
)
verl_args+=(
  data.shuffle=false
  data.train_max_samples=-1
  data.val_max_samples=256
  "+data.gen_batch_size=$train_batch"
  +algorithm.filter_groups.enable=true
  +algorithm.filter_groups.metric=acc
  "+algorithm.filter_groups.max_num_gen_batches=$max_num_gen_batches"
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  "+trainer.trace_auto_export_dir=$export_root/$run_name"
  +trainer.trace_checkpoint.enabled=true
  +trainer.trace_checkpoint.metric_key=val-core/trace_opsd_math/acc/mean@1
  +trainer.trace_checkpoint.best_margin=0.01
)
if [[ "$model_type" == "qwen3" ]]; then
  thinking_enabled=false
  [[ "$resolved_thinking_mode" == "on" ]] && thinking_enabled=true
  verl_args+=(
    "data.apply_chat_template_kwargs.enable_thinking=$thinking_enabled"
  )
fi
if [[ "$resume_mode" == "resume_path" ]]; then
  verl_args+=(trainer.resume_from_path="$resume_path")
fi
if [[ "$curriculum_requested" == true ]]; then
  curriculum_revalidate=false
  if [[ "$curriculum_action" == "revalidate" ]]; then
    curriculum_revalidate=true
  fi
  verl_args+=(
    "data.seed=$curriculum_seed"
    "+trainer.trace_curriculum.start_step=$curriculum_after"
    "+trainer.trace_curriculum.end_step=$curriculum_end"
    "+trainer.trace_curriculum.mid_per_step=$curriculum_mid"
    "+trainer.trace_curriculum.high_per_step=$curriculum_high"
    "+trainer.trace_curriculum.source_path=$curriculum_source"
    "+trainer.trace_curriculum.revalidate=$curriculum_revalidate"
  )
fi

set +e
if [[ "$curriculum_requested" == true ]]; then
  # A separate path avoids colliding with an outer user-provided tee command.
  env CUDA_VISIBLE_DEVICES="$training_devices" \
    bash "$repo_root/scripts/launch_training.sh" \
    --hardware-profile "$hardware_profile" \
    python -m "$trainer_module" "${verl_args[@]}" "${hydra_overrides[@]}" \
    2>&1 | tee -a "$run_dir/console-stage-$curriculum_after.log"
  training_pipe_status=("${PIPESTATUS[@]}")
  exit_code="${training_pipe_status[0]}"
  [[ "$exit_code" != "0" ]] || exit_code="${training_pipe_status[1]}"
else
  env CUDA_VISIBLE_DEVICES="$training_devices" \
    bash "$repo_root/scripts/launch_training.sh" \
    --hardware-profile "$hardware_profile" \
    python -m "$trainer_module" "${verl_args[@]}" "${hydra_overrides[@]}"
  exit_code="$?"
fi
set -e
if [[ "$exit_code" == "0" && "$revalidate_previous" == true ]]; then
  # The current process retains ownership of any tool service it started while
  # the next stage reuses that service; its EXIT trap releases it afterwards.
  set +e
  bash "$repo_root/scripts/run_grpo.sh" "${original_args[@]}"
  exit_code="$?"
  set -e
elif [[ "$curriculum_requested" == true && "$profile_name" == "train" && \
        -f "$run_dir/trace_validation_step_$curriculum_end.json" && \
        -f "$run_dir/global_step_$curriculum_end/manifest.json" ]]; then
  set +e
  package_trace_stage
  analysis_exit_code="$?"
  set -e
  if [[ "$analysis_exit_code" != "0" ]]; then
    printf 'TRACE checkpoint is saved; analysis bundle is incomplete or failed (exit %s).\n' \
      "$analysis_exit_code" >&2
    [[ "$exit_code" != "0" ]] || exit_code="$analysis_exit_code"
  fi
fi
exit "$exit_code"

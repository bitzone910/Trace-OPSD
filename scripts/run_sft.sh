#!/usr/bin/env bash
# 4 x A800 full SFT on the original Alpaca TIR data, then BF16 HF export.
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command_name=train
if [[ $# -gt 0 && "$1" != --* ]]; then command_name="$1"; shift; fi
assets_root=""
work_root=""
run_name=sft-qwen2.5-3b
runtime_config="$repo_root/configs/runtime/a800.yaml"
model_path=""
dataset_file=""
export_dir=""
checkpoint_root=""
forward_args=()

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_sft.sh [train|validate|export|inspect-checkpoint] [options]
  --assets-root DIR    Models/data root (default ../assets)
  --work-root DIR      External results/checkpoint root (default <assets>/runs)
  --run-name NAME      Default sft-qwen2.5-3b
  --model-path DIR     Default <assets>/models/Qwen2.5-3B-Instruct
  --dataset-file FILE  Default <assets>/datasets/sft/sft_math_tir.json
  --resume auto|none|latest|CHECKPOINT
  --export-dir DIR     Default <work>/exports/sft/<run>/final
  --checkpoint-root DIR  Advanced override for both training and HF export
Training uses 4 A800 GPUs, BF16/FSDP1, all 30,438 Alpaca records, cutoff 15000.
After training finishes the final checkpoint is exported to a complete BF16 HF
model. No validation split, best-loss model, or tool generation runs during SFT.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --assets-root|--work-root|--run-name|--model-path|--dataset-file|--export-dir|--runtime-config|--checkpoint-root)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      case "$1" in
        --assets-root) assets_root="$2" ;;
        --work-root) work_root="$2" ;;
        --run-name) run_name="$2" ;;
        --model-path) model_path="$2" ;;
        --dataset-file) dataset_file="$2" ;;
        --export-dir) export_dir="$2" ;;
        --runtime-config) runtime_config="$2" ;;
        --checkpoint-root) checkpoint_root="$2" ;;
      esac
      shift 2 ;;
    --hardware-profile)
      [[ $# -ge 2 && "$2" == a800 ]] || die "only the A800 profile is supported"
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) forward_args+=("$1"); shift ;;
  esac
done
case "$command_name" in train|validate|export|inspect-checkpoint) ;; *) usage; die "unknown command: $command_name" ;; esac
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || die "invalid run name"
source "$repo_root/scripts/assets_env.sh"
trace_opsd_assets_init "$assets_root" "$work_root"
assets_root="$TRACE_OPSD_ASSET_ROOT"
work_root="$TRACE_OPSD_WORK_ROOT"
model_path="${model_path:-$assets_root/models/Qwen2.5-3B-Instruct}"
dataset_file="${dataset_file:-$assets_root/datasets/sft/sft_math_tir.json}"
export_dir="${export_dir:-$work_root/exports/sft/$run_name/final}"
train_config="$repo_root/configs/sft/qwen2_5_3b_full_toolstar.yaml"
accelerate_config="$repo_root/configs/accelerate/fsdp1_4xa800.yaml"
export PYTHONPATH="$repo_root/frameworks/LlamaFactory/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$repo_root"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a available_devices <<<"$CUDA_VISIBLE_DEVICES"
else
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
  mapfile -t available_devices < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ')
fi
(( ${#available_devices[@]} >= 4 )) || die "SFT requires four visible A800 GPUs"
training_devices="$(IFS=,; printf '%s' "${available_devices[*]:0:4}")"
env CUDA_VISIBLE_DEVICES="$training_devices" conda run --no-capture-output -n trace-opsd \
  python -m trace_opsd.hardware_profile --profile a800 validate-devices --expected-count 4 >/dev/null
mkdir -p "$work_root/logs/sft" "$work_root/checkpoints/sft"
exec > >(tee -a "$work_root/logs/sft/$run_name-$command_name.log") 2>&1
printf 'SFT: %s\nData: %s\nWork root: %s\nGPUs: %s\n' "$run_name" "$dataset_file" "$work_root" "$training_devices"
common=(--assets-root "$assets_root" --work-root "$work_root" --runtime-config "$runtime_config")
if [[ -n "$checkpoint_root" ]]; then common+=(--checkpoint-root "$checkpoint_root"); fi
training=(--config "$train_config" --accelerate-config "$accelerate_config" --model-path "$model_path" --dataset-file "$dataset_file" "${common[@]}")
launch=(env CUDA_VISIBLE_DEVICES="$training_devices" bash "$repo_root/scripts/launch_training.sh" --hardware-profile a800)
case "$command_name" in
  validate)
    exec "${launch[@]}" python -m trace_opsd.sft.cli validate "${training[@]}" "${forward_args[@]}" ;;
  train)
    "${launch[@]}" python -m trace_opsd.sft.cli train --run-name "$run_name" "${training[@]}" "${forward_args[@]}" --preflight-only
    "${launch[@]}" accelerate launch --config_file "$accelerate_config" --num_processes 4 \
      -m trace_opsd.sft.cli train --run-name "$run_name" "${training[@]}" "${forward_args[@]}"
    exec "${launch[@]}" accelerate launch --config_file "$accelerate_config" --num_processes 4 \
      -m trace_opsd.sft.cli export --run-name "$run_name" --model-path "$model_path" \
      --checkpoint latest --target-dir "$export_dir" "${common[@]}" ;;
  export)
    exec "${launch[@]}" accelerate launch --config_file "$accelerate_config" --num_processes 4 \
      -m trace_opsd.sft.cli export --run-name "$run_name" --model-path "$model_path" \
      --target-dir "$export_dir" "${common[@]}" "${forward_args[@]}" ;;
  inspect-checkpoint)
    exec "${launch[@]}" python -m trace_opsd.sft.cli inspect-checkpoint --run-name "$run_name" "${common[@]}" "${forward_args[@]}" ;;
esac

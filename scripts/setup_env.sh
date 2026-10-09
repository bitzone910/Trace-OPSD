#!/usr/bin/env bash
# Install or validate the dedicated A800 environment.

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
env_name="trace-opsd"
env_file="${repo_root}/environment.yml"
validator="${repo_root}/scripts/verify_gpu_runtime.py"
profile_loader="${repo_root}/trace_opsd/hardware_profile.py"
vendor_verifier="${repo_root}/scripts/verify_vendored_framework.py"
native_auditor="${repo_root}/scripts/audit_native_artifacts.py"
cache_helper="${repo_root}/scripts/gpu_cache_env.sh"
verl_src="${TRACE_OPSD_VERL_SRC:-${repo_root}/frameworks/verl}"
llamafactory_src="${TRACE_OPSD_LLAMAFACTORY_SRC:-${repo_root}/frameworks/LlamaFactory}"

assets_root=""
work_root=""

expected_python="3.12.11"
expected_verl="9c38b8bb1876a81273d76de3e79328b2dd2b7b32"
expected_llamafactory="7af909522a951e3ad9f022ea6f88b6755257eaa5"
pypi_index="${TRACE_OPSD_PYPI_INDEX_URL:-https://pypi.org/simple}"
torch_index="${TRACE_OPSD_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

default_hardware_profile="a800"
hardware_profile="$default_hardware_profile"
mode="install"
requested_gpu_count=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --assets-root) assets_root="${2:?--assets-root requires a value}"; shift 2 ;;
    --work-root) work_root="${2:?--work-root requires a value}"; shift 2 ;;
    --hardware-profile)
      [[ $# -ge 2 ]] || { printf 'Error: --hardware-profile requires a value.\n' >&2; exit 2; }
      hardware_profile="$2"
      shift 2
      ;;
    --validate-only)
      [[ "$mode" == "install" ]] || { printf 'Error: installation modes are mutually exclusive.\n' >&2; exit 2; }
      mode="validate"
      shift
      ;;
    --preinstall-only)
      [[ "$mode" == "install" ]] || { printf 'Error: installation modes are mutually exclusive.\n' >&2; exit 2; }
      mode="preinstall"
      shift
      ;;
    --finalize)
      [[ "$mode" == "install" ]] || { printf 'Error: installation modes are mutually exclusive.\n' >&2; exit 2; }
      mode="finalize"
      shift
      ;;
    --gpu-count)
      [[ $# -ge 2 ]] || { printf 'Error: --gpu-count requires a value.\n' >&2; exit 2; }
      requested_gpu_count="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash scripts/setup_env.sh [--gpu-count N] [--assets-root DIR] [--work-root DIR]
  bash scripts/setup_env.sh --preinstall-only [--hardware-profile PROFILE] [--gpu-count N]
  bash scripts/setup_env.sh --finalize [--hardware-profile PROFILE] [--gpu-count N]
  bash scripts/setup_env.sh --validate-only [--hardware-profile PROFILE] [--gpu-count N]

The default mode creates/updates the dedicated `trace-opsd` Conda environment
for A800 GPUs (4 by default) and validates the actual CUDA kernels.
Models, data and wheels stay outside the source tree under --assets-root.
Caches, checkpoints and logs use --work-root (default: ASSETS/runs).
--preinstall-only installs the selected profile's dependencies without touching
the GPU, for preparing a portable image. --finalize consumes that preinstall on
the target GPU, performs validation, and records the final GPU-bound state.
Install and finalize also prepare the Python executor: a restricted account when
run as root, or your current UID/GID otherwise. Preinstall does not create it.
--validate-only never installs packages. The default profile is defined near the
top of this script and can be overridden without using exported variables.
EOF
      exit 0
      ;;
    *) printf 'Error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$hardware_profile" == "a800" ]] || { printf 'Error: only A800 is supported.\n' >&2; exit 2; }
source "$script_dir/assets_env.sh"
trace_opsd_assets_init "$assets_root" "$work_root"
state_root="${TRACE_OPSD_WORK_ROOT}/environment-state"
state_file="${state_root}/trace-opsd-install.json"
preinstall_file="${state_root}/trace-opsd-preinstall.json"
freeze_file="${state_root}/trace-opsd-pip-freeze.txt"
report_root="${state_root}/gpu-reports"

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

version_ge() {
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}

command -v conda >/dev/null 2>&1 || die "conda is required"
command -v python >/dev/null 2>&1 || die "python is required for bootstrap checks"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
[[ "$(uname -s)" == "Linux" ]] || die "Trace-OPSD requires Linux"
[[ "$(uname -m)" == "x86_64" ]] || die "Trace-OPSD wheels require Linux x86_64"

[[ -f "$profile_loader" ]] || die "missing hardware profile loader: $profile_loader"
profile_value() {
  python "$profile_loader" --profile "$hardware_profile" get "$1"
}
profile_value name >/dev/null || die "invalid hardware profile: $hardware_profile"
requirements_file="$(profile_value dependency_lock)"
lock_file="$requirements_file"
dependency_input="$(profile_value dependency_input)"
runtime_config="$(profile_value runtime_config)"
workload="$(profile_value workload)"
vendored_frameworks="$(profile_value vendored_frameworks)"
vllm_attention_backend="$(profile_value inference.vllm_attention_backend)"
xformers_kernel_policy="$(profile_value inference.xformers_kernel_policy)"
minimum_driver="$(profile_value gpu.minimum_driver)"
expected_cache_arch="$(profile_value gpu.cache_arch)"
torch_cuda_arch_list="$(profile_value gpu.torch_cuda_arch_list)"
final_gpu_count="$(profile_value gpu.final_gpu_count)"
distributed_gpu_counts="$(profile_value gpu.distributed_gpu_counts)"
flash_version="$(profile_value flash_attention.version)"
flash_filename="$(profile_value flash_attention.filename)"
flash_url="$(profile_value flash_attention.url)"
flash_sha256="$(profile_value flash_attention.sha256)"
flash_size_bytes="$(profile_value flash_attention.size_bytes)"
if [[ -z "$requested_gpu_count" ]]; then
  requested_gpu_count="$final_gpu_count"
fi
case ",$distributed_gpu_counts," in
  *,"$requested_gpu_count",*) ;;
  *) die "--gpu-count must be one of $distributed_gpu_counts for $hardware_profile" ;;
esac

for required in \
  "$env_file" "$dependency_input" "$requirements_file" "$lock_file" "$runtime_config" "$validator" \
  "$vendor_verifier" "$native_auditor" "$cache_helper"; do
  [[ -f "$required" ]] || die "missing required file: $required"
done
[[ -d "$verl_src" ]] || die "missing vendored verl source: $verl_src"
if [[ ",$vendored_frameworks," == *,llamafactory,* ]]; then
  [[ -d "$llamafactory_src" ]] || die "missing vendored LlamaFactory source: $llamafactory_src"
fi

mkdir -p "$state_root" "$report_root"
cd -- "$repo_root"

env_exists="$(
  { conda env list --json || true; } | python -c '
import json, os, sys
name = sys.argv[1]
try:
    envs = json.load(sys.stdin).get("envs", [])
except json.JSONDecodeError:
    envs = []
print("yes" if any(os.path.basename(path.rstrip(os.sep)) == name for path in envs) else "no")
' "$env_name"
)"

if [[ "$mode" != "validate" && -f "$state_file" ]]; then
  installed_profile="$(python - "$state_file" <<'PY'
import json
import sys

try:
    state = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    print("")
else:
    print(state.get("hardware_profile") or "")
PY
)"
  if [[ -n "$installed_profile" && "$installed_profile" != "$hardware_profile" ]]; then
    die "existing trace-opsd environment is bound to $installed_profile; create a fresh environment for $hardware_profile"
  fi
fi

if [[ "$mode" == "install" || "$mode" == "preinstall" ]]; then
  if [[ "$env_exists" == "yes" ]]; then
    current_python="$(conda run --no-capture-output -n "$env_name" python -c 'import platform; print(platform.python_version())')"
    [[ "$current_python" == "$expected_python" ]] || die \
      "existing $env_name uses Python $current_python; expected $expected_python"
    conda env update --name "$env_name" --file "$env_file"
  else
    conda env create --file "$env_file"
  fi
else
  [[ "$env_exists" == "yes" ]] || die \
    "dedicated environment '$env_name' is missing; $mode never installs it"
fi

run_env() {
  conda run --no-capture-output -n "$env_name" "$@"
}

env_python="$(run_env python -c 'import sys; print(sys.executable)')"
[[ -x "$env_python" ]] || die "cannot resolve the $env_name Python interpreter"
python_version="$(run_env python -c 'import platform; print(platform.python_version())')"
[[ "$python_version" == "$expected_python" ]] || die \
  "$env_name must use Python $expected_python, found $python_version"

verl_digest="$(
  python "$vendor_verifier" "$verl_src" \
    --expected-name verl --expected-revision "$expected_verl" --print-digest
)" || die "vendored verl verification failed"
llamafactory_digest="$(
  if [[ ",$vendored_frameworks," == *,llamafactory,* ]]; then
    python "$vendor_verifier" "$llamafactory_src" \
      --expected-name LlamaFactory --expected-revision "$expected_llamafactory" \
      --print-digest
  fi
)" || die "vendored LlamaFactory verification failed"
native_sources=(--source "$verl_src")
if [[ ",$vendored_frameworks," == *,llamafactory,* ]]; then
  native_sources+=(--source "$llamafactory_src")
fi
python "$native_auditor" "${native_sources[@]}" || \
  die "vendored framework source contains native artifacts"

if [[ "$mode" == "install" || "$mode" == "preinstall" ]]; then
  run_env python -m pip install --upgrade \
    --index-url "$pypi_index" \
    'pip==25.1.1' 'setuptools==79.0.1' 'wheel==0.45.1'
  run_env python -m pip install --upgrade --require-hashes \
    --index-url "$pypi_index" --extra-index-url "$torch_index" \
    --requirement "$requirements_file"

  # FlashAttention is an external, hash-locked artifact; do not compile it here.
  flash_wheel="${TRACE_OPSD_ASSET_ROOT}/wheels/${flash_filename}"
  flash_source="$flash_wheel"
  [[ -f "$flash_wheel" ]] || die \
    "missing profile wheel: $flash_wheel (source: $flash_url)"
  printf 'Using local FlashAttention wheel: %s\n' "$flash_wheel"
  [[ "$(basename -- "$flash_wheel")" == "$flash_filename" ]] || die \
    "unexpected FlashAttention wheel filename: $flash_wheel"
  [[ "$(wc -c <"$flash_wheel" | tr -d ' ')" == "$flash_size_bytes" ]] || die \
    "FlashAttention wheel size mismatch: $flash_wheel"
  printf '%s  %s\n' "$flash_sha256" "$flash_wheel" | sha256sum --check --status || die \
    "FlashAttention wheel digest mismatch: $flash_wheel"
  run_env python -m pip install --force-reinstall --no-deps --no-index "$flash_wheel"

  # These two distributions are Python-only. Dependencies are already locked;
  # disabling cache/build isolation prevents hidden dependency or wheel reuse.
  run_env python -m pip install --force-reinstall --no-deps \
    --no-build-isolation --no-cache-dir "$verl_src"
  if [[ ",$vendored_frameworks," == *,llamafactory,* ]]; then
    run_env python -m pip install --force-reinstall --no-deps \
      --no-build-isolation --no-cache-dir "$llamafactory_src"
  fi
  run_env python -m pip install --no-deps --no-build-isolation --editable "$repo_root"
fi

native_distributions=(--distribution verl)
if [[ ",$vendored_frameworks," == *,llamafactory,* ]]; then
  native_distributions+=(--distribution llamafactory)
fi
run_env python "$native_auditor" "${native_distributions[@]}" || \
  die "installed vendored framework contains native artifacts"

flash_direct_url="$(run_env python - "$flash_version" "$flash_sha256" <<'PY'
import json
import sys
from importlib import metadata

wanted_version, wanted_sha256 = sys.argv[1:]
distribution = metadata.distribution("flash-attn")
if distribution.version != wanted_version:
    raise RuntimeError(
        f"unexpected flash-attn version: {distribution.version} != {wanted_version}"
    )
payload = distribution.read_text("direct_url.json")
if payload is None:
    raise RuntimeError("flash-attn has no PEP 610 direct_url.json")
direct = json.loads(payload)
url = direct.get("url", "")
archive = direct.get("archive_info") or {}
hashes = archive.get("hashes") or {}
found_hash = hashes.get("sha256")
if found_hash is None and str(archive.get("hash", "")).startswith("sha256="):
    found_hash = archive["hash"].split("=", 1)[1]
if found_hash != wanted_sha256:
    raise RuntimeError(
        f"flash-attn digest mismatch: expected {wanted_sha256}, found {found_hash}"
    )
print(url)
PY
)"
printf 'Verified FlashAttention wheel artifact: sha256=%s source=%s\n' \
  "$flash_sha256" "$flash_direct_url"

if [[ "$mode" == "validate" && ! -f "$state_file" ]]; then
  printf 'Installation state is absent; validating the live environment and GPU runtime.\n'
fi

lock_sha256="$(sha256sum "$lock_file" | awk '{print $1}')"
requirements_sha256="$(sha256sum "$dependency_input" | awk '{print $1}')"
if [[ "$mode" == "preinstall" ]]; then
  run_env python - "$preinstall_file" "$hardware_profile" "$expected_python" \
    "$lock_sha256" "$requirements_sha256" "$verl_digest" "$llamafactory_digest" \
    "$flash_version" "$flash_sha256" "$workload" "$vendored_frameworks" \
    "$vllm_attention_backend" "$xformers_kernel_policy" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    path,
    hardware_profile,
    python_version,
    lock_sha256,
    requirements_sha256,
    verl_sha256,
    llamafactory_sha256,
    flash_version,
    flash_sha256,
    workload,
    vendored_frameworks,
    vllm_attention_backend,
    xformers_kernel_policy,
) = sys.argv[1:]
state = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "environment": "trace_opsd",
    "hardware_profile": hardware_profile,
    "workload": workload,
    "vendored_frameworks": vendored_frameworks.split(","),
    "python_version": python_version,
    "lock_sha256": lock_sha256,
    "requirements_sha256": requirements_sha256,
    "verl": {"source_sha256": verl_sha256},
    "flash_attn": {"version": flash_version, "sha256": flash_sha256},
    "attention": {
        "vllm_backend": vllm_attention_backend,
        "xformers_kernel_policy": xformers_kernel_policy,
    },
}
if llamafactory_sha256:
    state["llamafactory"] = {"source_sha256": llamafactory_sha256}
target = Path(path)
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
PY
  printf '%s\n' \
    "Trace-OPSD dependencies were preinstalled without GPU validation." \
    "Target hardware profile: $hardware_profile" \
    "Preinstall state: $preinstall_file" \
    "Run --finalize on the target GPU before using this environment."
  exit 0
fi

if [[ "$mode" == "finalize" ]]; then
  [[ -f "$preinstall_file" ]] || die \
    "missing preinstall state for $hardware_profile: $preinstall_file"
  run_env python - "$preinstall_file" "$hardware_profile" "$expected_python" \
    "$lock_sha256" "$requirements_sha256" "$verl_digest" "$llamafactory_digest" \
    "$flash_version" "$flash_sha256" "$workload" "$vendored_frameworks" \
    "$vllm_attention_backend" "$xformers_kernel_policy" <<'PY'
import json
import sys

(
    path,
    hardware_profile,
    python_version,
    lock_sha256,
    requirements_sha256,
    verl_sha256,
    llamafactory_sha256,
    flash_version,
    flash_sha256,
    workload,
    vendored_frameworks,
    vllm_attention_backend,
    xformers_kernel_policy,
) = sys.argv[1:]
state = json.load(open(path, encoding="utf-8"))
expected = {
    "environment": "trace_opsd",
    "hardware_profile": hardware_profile,
    "workload": workload,
    "vendored_frameworks": vendored_frameworks.split(","),
    "python_version": python_version,
    "lock_sha256": lock_sha256,
    "requirements_sha256": requirements_sha256,
    "verl_sha256": verl_sha256,
    "llamafactory_sha256": llamafactory_sha256,
    "flash_version": flash_version,
    "flash_sha256": flash_sha256,
    "vllm_attention_backend": vllm_attention_backend,
    "xformers_kernel_policy": xformers_kernel_policy,
}
actual = {
    "environment": state.get("environment"),
    "hardware_profile": state.get("hardware_profile"),
    "workload": state.get("workload"),
    "vendored_frameworks": state.get("vendored_frameworks"),
    "python_version": state.get("python_version"),
    "lock_sha256": state.get("lock_sha256"),
    "requirements_sha256": state.get("requirements_sha256"),
    "verl_sha256": (state.get("verl") or {}).get("source_sha256"),
    "llamafactory_sha256": (state.get("llamafactory") or {}).get("source_sha256") or "",
    "flash_version": (state.get("flash_attn") or {}).get("version"),
    "flash_sha256": (state.get("flash_attn") or {}).get("sha256"),
    "vllm_attention_backend": (state.get("attention") or {}).get("vllm_backend"),
    "xformers_kernel_policy": (state.get("attention") or {}).get("xformers_kernel_policy"),
}
for key, wanted in expected.items():
    if actual[key] != wanted:
        raise RuntimeError(f"stale preinstall state for {key}: {actual[key]!r} != {wanted!r}")
print("Preinstall state matches the repository inputs.")
PY
fi
if [[ "$mode" == "validate" && -f "$state_file" ]]; then
  run_env python -m trace_opsd.hardware_profile --profile "$hardware_profile" \
    validate-state --state "$state_file" || die \
    "installed environment does not match $hardware_profile"
  run_env python - "$state_file" "$lock_sha256" "$requirements_sha256" \
    "$verl_digest" "$llamafactory_digest" "$hardware_profile" "$workload" \
    "$vendored_frameworks" <<'PY'
import json
import sys

(
    path,
    lock_sha,
    requirements_sha,
    verl_sha,
    llamafactory_sha,
    hardware_profile,
    workload,
    vendored_frameworks,
) = sys.argv[1:]
state = json.load(open(path, encoding="utf-8"))
expected = {
    "environment": "trace_opsd",
    "hardware_profile": hardware_profile,
    "workload": workload,
    "vendored_frameworks": vendored_frameworks.split(","),
    "lock_sha256": lock_sha,
    "requirements_sha256": requirements_sha,
    "verl_sha256": verl_sha,
    "llamafactory_sha256": llamafactory_sha,
}
actual = {
    "environment": state.get("environment"),
    "hardware_profile": state.get("hardware_profile"),
    "workload": state.get("workload"),
    "vendored_frameworks": state.get("vendored_frameworks"),
    "lock_sha256": state.get("lock_sha256"),
    "requirements_sha256": state.get("requirements_sha256"),
    "verl_sha256": (state.get("verl") or {}).get("source_sha256"),
    "llamafactory_sha256": (state.get("llamafactory") or {}).get("source_sha256") or "",
}
for key, wanted in expected.items():
    if actual[key] != wanted:
        raise RuntimeError(f"stale cloned environment state for {key}: {actual[key]!r} != {wanted!r}")
print("Installed environment state matches the repository inputs.")
PY
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required for $mode"
mapfile -t drivers < <(
  nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | tr -d ' '
)
(( ${#drivers[@]} > 0 )) || die "nvidia-smi reports no visible GPU"
for driver in "${drivers[@]}"; do
  version_ge "$driver" "$minimum_driver" || die \
    "NVIDIA driver $driver is older than required $minimum_driver"
done

export TRACE_OPSD_CACHE_PYTHON="$env_python"
source "$cache_helper"
trace_opsd_configure_gpu_caches "$expected_cache_arch" "$torch_cuda_arch_list"
export VLLM_ATTENTION_BACKEND="$vllm_attention_backend"
expected_gpu="$hardware_profile"

timestamp="$(date -u +'%Y%m%dT%H%M%SZ')"
report_dir="${report_root}/${timestamp}"
mkdir -p "$report_dir"

# BEGIN python executor setup
executor_status="not-provisioned-validate-only"
executor_user=""
executor_uid=""
executor_gid=""
executor_log=""
if [[ "$mode" == "install" || "$mode" == "finalize" ]]; then
  executor_log="${report_dir}/python-executor-setup.log"
  if [[ "$(id -u)" -eq 0 ]]; then
    executor_user="trace-opsd-exec"
    if ! command -v setfacl >/dev/null 2>&1; then
      command -v apt-get >/dev/null 2>&1 || die \
        "Python executor requires setfacl; install your distribution's acl package (Debian/Ubuntu: apt-get install -y acl), then rerun setup"
      printf 'Installing acl for the restricted Python executor.\n'
      if ! DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y acl \
        >"$executor_log" 2>&1; then
        cat "$executor_log" >&2
        die "could not install acl; run apt-get update && apt-get install -y acl, then rerun setup"
      fi
      command -v setfacl >/dev/null 2>&1 || die \
        "acl installation did not provide setfacl; prepare setfacl and rerun setup"
    fi
    if ! bash "$repo_root/scripts/setup_executor_user.sh" "$executor_user" "$env_python" \
      >>"$executor_log" 2>&1; then
      cat "$executor_log" >&2
      die "restricted Python executor setup failed; see $executor_log"
    fi
    executor_uid="$(id -u "$executor_user")"
    executor_gid="$(id -g "$executor_user")"
    executor_status="restricted-user-ready"
  else
    executor_user="$(id -un)"
    executor_uid="$(id -u)"
    executor_gid="$(id -g)"
    executor_status="current-user-ready"
    printf 'Using current user %s (%s:%s).\n' \
      "$executor_user" "$executor_uid" "$executor_gid" >"$executor_log"
  fi
  [[ "$executor_uid" =~ ^[0-9]+$ && "$executor_uid" -gt 0 && \
     "$executor_gid" =~ ^[0-9]+$ && "$executor_gid" -gt 0 ]] || die \
    "Python executor must have nonzero numeric UID/GID; got ${executor_uid}:${executor_gid}"
  export EXECUTOR_UID="$executor_uid" EXECUTOR_GID="$executor_gid"
  printf 'Python executor ready: %s (%s:%s).\n' \
    "$executor_user" "$executor_uid" "$executor_gid"
fi
# END python executor setup

nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv,noheader >"${report_dir}/nvidia-smi.csv"

CUDA_VISIBLE_DEVICES=0 run_env python "$validator" single \
  --expected-gpu "$expected_gpu" --output "${report_dir}/single-gpu.json"

gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[[ "$gpu_count" -eq "$requested_gpu_count" ]] || die \
  "$hardware_profile validation requested $requested_gpu_count GPUs; found $gpu_count"
run_env python -m trace_opsd.hardware_profile --profile "$hardware_profile" \
  validate-devices --expected-count "$requested_gpu_count" >/dev/null

distributed_status="not-required-single-gpu"
if (( requested_gpu_count > 1 )); then
  visible_devices="$(seq -s, 0 $((requested_gpu_count - 1)))"
  CUDA_VISIBLE_DEVICES="$visible_devices" run_env torchrun \
    --standalone --nnodes=1 --nproc-per-node="$requested_gpu_count" \
    "$validator" distributed --expected-gpu "$hardware_profile" \
    --output "${report_dir}/distributed.json"
  if [[ "$requested_gpu_count" -eq "$final_gpu_count" ]]; then
    distributed_status="passed-final-${requested_gpu_count}gpu"
  else
    distributed_status="passed-preliminary-${requested_gpu_count}gpu"
  fi
fi

model_status="not-run-no-local-model-configured"
if [[ -n "${TRACE_OPSD_VALIDATION_MODEL:-}" ]]; then
  [[ -d "$TRACE_OPSD_VALIDATION_MODEL" ]] || die \
    "TRACE_OPSD_VALIDATION_MODEL must be an existing local directory"
  set +e
  CUDA_VISIBLE_DEVICES=0 run_env python "$validator" vllm-model \
    --model "$TRACE_OPSD_VALIDATION_MODEL" \
    --output "${report_dir}/vllm-model.json" 2>&1 | \
    tee "${report_dir}/vllm-model.log"
  vllm_status="${PIPESTATUS[0]}"
  set -e
  [[ "$vllm_status" -eq 0 ]] || die "vLLM model probe failed"
  if grep -Fq "Using XFormers backend" "${report_dir}/vllm-model.log"; then
    die "vLLM selected the forbidden XFormers backend"
  fi
  model_status="passed"
fi

run_env python -m pip freeze --all >"$freeze_file"
export TRACE_OPSD_SETUP_MODE="$mode"
export TRACE_OPSD_HARDWARE_PROFILE="$hardware_profile"
export TRACE_OPSD_WORKLOAD="$workload"
export TRACE_OPSD_VENDORED_FRAMEWORKS="$vendored_frameworks"
export TRACE_OPSD_STATE_FILE="$state_file"
export TRACE_OPSD_REPORT_DIR="$report_dir"
export TRACE_OPSD_LOCK_SHA256="$lock_sha256"
export TRACE_OPSD_REQUIREMENTS_SHA256="$requirements_sha256"
export TRACE_OPSD_VERL_SHA256="$verl_digest"
export TRACE_OPSD_LLAMAFACTORY_SHA256="$llamafactory_digest"
export TRACE_OPSD_PYTHON_VERSION="$python_version"
export TRACE_OPSD_DISTRIBUTED_STATUS="$distributed_status"
export TRACE_OPSD_MODEL_STATUS="$model_status"
export TRACE_OPSD_GPU_CACHE_ARCH="$expected_cache_arch"
export TRACE_OPSD_DEPENDENCY_LOCK="$requirements_file"
export TRACE_OPSD_DEPENDENCY_INPUT="$dependency_input"
export TRACE_OPSD_VLLM_ATTENTION_BACKEND="$vllm_attention_backend"
export TRACE_OPSD_XFORMERS_KERNEL_POLICY="$xformers_kernel_policy"
export TRACE_OPSD_FLASH_ATTN_VERSION="$flash_version"
export TRACE_OPSD_FLASH_ATTN_SHA256="$flash_sha256"
export TRACE_OPSD_FLASH_ATTN_SOURCE="$flash_direct_url"
export TRACE_OPSD_EXECUTOR_STATUS="$executor_status"
export TRACE_OPSD_EXECUTOR_USER="$executor_user"
export TRACE_OPSD_EXECUTOR_UID="$executor_uid"
export TRACE_OPSD_EXECUTOR_GID="$executor_gid"
export TRACE_OPSD_EXECUTOR_LOG="$executor_log"
run_env python - <<'PY'
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

report_dir = Path(os.environ["TRACE_OPSD_REPORT_DIR"])
hardware = []
with (report_dir / "nvidia-smi.csv").open(encoding="utf-8") as handle:
    for row in csv.reader(handle):
        hardware.append(
            {
                "index": row[0].strip(),
                "name": row[1].strip(),
                "uuid": row[2].strip(),
                "driver_version": row[3].strip(),
                "memory_total": row[4].strip(),
            }
        )
single = json.loads((report_dir / "single-gpu.json").read_text(encoding="utf-8"))
distributed_path = report_dir / "distributed.json"
distributed = (
    json.loads(distributed_path.read_text(encoding="utf-8"))
    if distributed_path.exists()
    else {"status": os.environ["TRACE_OPSD_DISTRIBUTED_STATUS"]}
)
model_path = report_dir / "vllm-model.json"
model = (
    json.loads(model_path.read_text(encoding="utf-8"))
    if model_path.exists()
    else {"status": os.environ["TRACE_OPSD_MODEL_STATUS"]}
)
training_status = (
    f"{os.environ['TRACE_OPSD_HARDWARE_PROFILE']}-"
    f"{os.environ['TRACE_OPSD_DISTRIBUTED_STATUS']}-training-validation-not-run"
)
python_executor = {
    "status": os.environ["TRACE_OPSD_EXECUTOR_STATUS"],
    "user": os.environ["TRACE_OPSD_EXECUTOR_USER"] or None,
    "uid": int(os.environ["TRACE_OPSD_EXECUTOR_UID"]) if os.environ["TRACE_OPSD_EXECUTOR_UID"] else None,
    "gid": int(os.environ["TRACE_OPSD_EXECUTOR_GID"]) if os.environ["TRACE_OPSD_EXECUTOR_GID"] else None,
    "log": os.environ["TRACE_OPSD_EXECUTOR_LOG"] or None,
}
report = {
    "schema_version": 4,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "environment": "trace_opsd",
    "hardware_profile": os.environ["TRACE_OPSD_HARDWARE_PROFILE"],
    "workload": os.environ["TRACE_OPSD_WORKLOAD"],
    "vendored_frameworks": os.environ["TRACE_OPSD_VENDORED_FRAMEWORKS"].split(","),
    "mode": os.environ["TRACE_OPSD_SETUP_MODE"],
    "status": training_status,
    "lock_sha256": os.environ["TRACE_OPSD_LOCK_SHA256"],
    "requirements_sha256": os.environ["TRACE_OPSD_REQUIREMENTS_SHA256"],
    "hardware": hardware,
    "cache_paths": single["cache_paths"],
    "single_gpu": single,
    "distributed": distributed,
    "vllm_model": model,
    "training_validation": {"sft": "not-run", "trace_opsd": "not-run"},
    "python_executor": python_executor,
    "attention": {
        "vllm_backend": os.environ["TRACE_OPSD_VLLM_ATTENTION_BACKEND"],
        "xformers_kernel_policy": os.environ["TRACE_OPSD_XFORMERS_KERNEL_POLICY"],
    },
    "dependency_input": os.environ["TRACE_OPSD_DEPENDENCY_INPUT"],
    "dependency_lock": os.environ["TRACE_OPSD_DEPENDENCY_LOCK"],
    "gpu_cache_arch": os.environ["TRACE_OPSD_GPU_CACHE_ARCH"],
    "flash_attn": {"version": os.environ["TRACE_OPSD_FLASH_ATTN_VERSION"], "sha256": os.environ["TRACE_OPSD_FLASH_ATTN_SHA256"], "source": os.environ["TRACE_OPSD_FLASH_ATTN_SOURCE"]},
}
(report_dir / "report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

state_path = Path(os.environ["TRACE_OPSD_STATE_FILE"])
state = {
    "schema_version": 6,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "environment": "trace_opsd",
    "hardware_profile": os.environ["TRACE_OPSD_HARDWARE_PROFILE"],
    "workload": os.environ["TRACE_OPSD_WORKLOAD"],
    "vendored_frameworks": os.environ["TRACE_OPSD_VENDORED_FRAMEWORKS"].split(","),
    "gpu_cache_arch": os.environ["TRACE_OPSD_GPU_CACHE_ARCH"],
    "dependency_lock": os.environ["TRACE_OPSD_DEPENDENCY_LOCK"],
    "dependency_input": os.environ["TRACE_OPSD_DEPENDENCY_INPUT"],
    "python_version": os.environ["TRACE_OPSD_PYTHON_VERSION"],
    "lock_sha256": os.environ["TRACE_OPSD_LOCK_SHA256"],
    "requirements_sha256": os.environ["TRACE_OPSD_REQUIREMENTS_SHA256"],
    "verl": {"revision": "9c38b8bb1876a81273d76de3e79328b2dd2b7b32", "source_sha256": os.environ["TRACE_OPSD_VERL_SHA256"]},
    "attention": {"vllm_backend": os.environ["TRACE_OPSD_VLLM_ATTENTION_BACKEND"], "xformers_kernel_policy": os.environ["TRACE_OPSD_XFORMERS_KERNEL_POLICY"]},
    "flash_attn": {"version": os.environ["TRACE_OPSD_FLASH_ATTN_VERSION"], "sha256": os.environ["TRACE_OPSD_FLASH_ATTN_SHA256"], "source": os.environ["TRACE_OPSD_FLASH_ATTN_SOURCE"]},
    "last_report": str(report_dir / "report.json"),
    "python_executor": python_executor,
}
if os.environ["TRACE_OPSD_LLAMAFACTORY_SHA256"]:
    state["llamafactory"] = {"revision": "7af909522a951e3ad9f022ea6f88b6755257eaa5", "source_sha256": os.environ["TRACE_OPSD_LLAMAFACTORY_SHA256"]}
temporary = state_path.with_suffix(".tmp")
temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(state_path)
PY

printf '%s\n' \
  "Trace-OPSD environment $mode validation passed." \
  "Environment: $env_name" \
  "Hardware profile: $hardware_profile" \
  "Report: ${report_dir}/report.json"

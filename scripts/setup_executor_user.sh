#!/usr/bin/env bash
# Provision a restricted Linux user that can import from the active Conda env.

set -euo pipefail

executor_user="${1:-toolelite-exec}"
python_bin="${2:-${CONDA_PREFIX:-}/bin/python}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Error: run this script as root." >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" && $# -lt 2 ]]; then
  echo "Error: activate the target Conda environment first." >&2
  exit 1
fi

if [[ ! -x "$python_bin" ]]; then
  echo "Error: Python executable does not exist: $python_bin" >&2
  exit 1
fi

if ! command -v setfacl >/dev/null 2>&1; then
  echo "Error: setfacl is required. Install it with: apt-get install -y acl" >&2
  exit 1
fi

if ! command -v runuser >/dev/null 2>&1; then
  echo "Error: runuser is required to verify the restricted account." >&2
  exit 1
fi

python_bin="$(readlink -f "$python_bin")"
environment_dir="$(dirname "$(dirname "$python_bin")")"

if ! id "$executor_user" >/dev/null 2>&1; then
  useradd \
    --system \
    --no-create-home \
    --shell /usr/sbin/nologin \
    "$executor_user"
fi

# Grant traverse-only ACLs on environment ancestors (for example /root), then
# read/execute access to the selected Conda environment itself.
ancestor="$(dirname "$environment_dir")"
while [[ "$ancestor" != "/" ]]; do
  setfacl -m "u:${executor_user}:--x" "$ancestor"
  ancestor="$(dirname "$ancestor")"
done
setfacl -R -m "u:${executor_user}:rX" "$environment_dir"

runuser -u "$executor_user" -- \
  "$python_bin" -I -c "import sys; print(sys.executable)"

if "$python_bin" -I -c "import numpy" >/dev/null 2>&1; then
  runuser -u "$executor_user" -- \
    "$python_bin" -I -c "import numpy; print(numpy.__version__)"
fi

executor_uid="$(id -u "$executor_user")"
executor_gid="$(id -g "$executor_user")"

echo "Executor user is ready: ${executor_user} (${executor_uid}:${executor_gid})"
echo "export EXECUTOR_UID=${executor_uid}"
echo "export EXECUTOR_GID=${executor_gid}"

#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash utility_analysis/scripts/run_compute_utilities_roles_cluster.sh <model_key> <hf_repo_id>
# Roles, options_path, save_dir, etc. come from experiments.yaml (compute_utilities_roles).
# Example:
#   bash utility_analysis/scripts/run_compute_utilities_roles_cluster.sh \
#     llama-31-8b-instruct \
#     RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8

MODEL_KEY="${1:-}"
HF_REPO_ID="${2:-}"

if [[ -z "${MODEL_KEY}" || -z "${HF_REPO_ID}" ]]; then
  echo "Usage: $0 <model_key> <hf_repo_id>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTIL_ANALYSIS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${UTIL_ANALYSIS_DIR}/.." && pwd)"

VENV_ACTIVATE="${REPO_ROOT}/venv/bin/activate"
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Virtual environment not found at ${VENV_ACTIVATE}"
  exit 1
fi
source "${VENV_ACTIVATE}"

SCRATCH_BASE="/scratch/${USER}"
MODEL_DIR="${SCRATCH_BASE}/models/${MODEL_KEY}"
mkdir -p "${MODEL_DIR}"

echo "Downloading ${HF_REPO_ID} to ${MODEL_DIR}"
python - <<'PY' "${HF_REPO_ID}" "${MODEL_DIR}"
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
PY

cd "${UTIL_ANALYSIS_DIR}"
python run_experiments.py \
  --experiments compute_utilities_roles \
  --models "${MODEL_KEY}" \
  --overwrite_results

echo "Done. Results dir is set by experiments.yaml (compute_utilities_roles.save_dir)."
deactivate
exit 0

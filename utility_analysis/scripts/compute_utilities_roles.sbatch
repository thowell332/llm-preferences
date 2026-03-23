#!/bin/bash -l
#SBATCH -J compute_utils_roles_gpu
#SBATCH --time=00-06:00:00
#SBATCH -p gpu,preempt
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=32g
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="a100-80G"
#SBATCH --output=compute_utils_roles_gpu.%j.%N.out
#SBATCH --error=compute_utils_roles_gpu.%j.%N.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=thomas.howell@tufts.edu

set -euo pipefail

module purge
module load miniforge/25.3.0
module load cuda/12.9.0

# Customize these:
MODEL_KEY="${MODEL_KEY:-llama-31-8b-instruct}"
HF_REPO_ID="${HF_REPO_ID:-RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8}"

REPO_ROOT="${SLURM_SUBMIT_DIR}"
bash "${REPO_ROOT}/utility_analysis/scripts/run_compute_utilities_roles_cluster.sh" "${MODEL_KEY}" "${HF_REPO_ID}"

#!/usr/bin/env bash
#SBATCH --job-name=infodfa_mixer
#SBATCH --partition=kempner_eng
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# MLP-Mixer conditioned-DFA test (pre-registered P5 in PREDICTIONS.md):
# does activity-side conditioning survive the token/channel factorization
# of mixer-family architectures, and how much of the gain does LayerNorm
# absorb? Grid: label-noise {0, 0.4} x layernorm {on, off}; each task runs
# bp / dfa_random / ndfa_random / ndfa_random_kronecker at the runner's
# paper-matched defaults (dim 128, depth 4, patch 4, 14 epochs, lr 0.04,
# 5 data seeds x 3 feedback seeds).
set -euo pipefail

# Optionally activate a conda env by exporting CONDA_ENV before submitting;
# otherwise the environment active at submission time is used.
if [ -n "${CONDA_ENV:-}" ]; then
    source "${CONDA_SH:-/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV}"
fi

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
OUT_ROOT="${OUT_ROOT:-results/infodfa_mixer_v1}"

LABEL_NOISES=(0.0 0.4)
LN_FLAGS=(--layernorm --no-layernorm)
LN_TAGS=(ln noln)

NOISE_INDEX=$((TASK_ID / 2))
LN_INDEX=$((TASK_ID % 2))
LABEL_NOISE="${LABEL_NOISES[$NOISE_INDEX]}"
LN_FLAG="${LN_FLAGS[$LN_INDEX]}"
LN_TAG="${LN_TAGS[$LN_INDEX]}"
noise_tag="${LABEL_NOISE//./p}"

OUT_DIR="${OUT_ROOT}/label_${noise_tag}_${LN_TAG}"
mkdir -p "${OUT_DIR}"

python experiments/run_dfa_mixer.py \
    --dataset cifar10 \
    --download \
    --label-noise "${LABEL_NOISE}" \
    "${LN_FLAG}" \
    --cuda \
    --output-dir "${OUT_DIR}"

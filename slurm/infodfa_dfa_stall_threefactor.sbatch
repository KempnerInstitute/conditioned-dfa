#!/usr/bin/env bash
# Validation-selected activity/error/K-nDFA experiment on the DFA-Stall MNIST setup.
# Tasks 0--5: activity-damping development grid.
# Tasks 6--11: error-damping development grid.
# Task 12: raw DFA and K-nDFA at the independently selected factors.
# Tasks 13--15: frozen confirmation, one task per feedback seed.
#SBATCH --job-name=infodfa_3factor
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-15%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
DAMPINGS=(0.03 0.1 0.3 1 3 10)
COMMON=(
  --total-steps 1000
  --batch-size 128
  --hidden 300
  --lr 1e-3
  --norm-match-hidden
  --device cuda
  --validation-size 5000
  --validation-seed 12345
  --probe-n 5000
  --feature-probe-n 256
  --eval-every 50
  --metric-every 10
  --no-plots
)

if (( TASK_ID < 6 )); then
  DAMPING="${DAMPINGS[$TASK_ID]}"
  TAG="${DAMPING/./p}"
  python experiments/run_dfa_stall_comparison.py \
    --output-dir "results/dfa_stall_threefactor_dev_v1/activity_d${TAG}" \
    --methods ndfa --seeds 42 43 44 --feedback-seeds 0 \
    --damping "${DAMPING}" "${COMMON[@]}"
elif (( TASK_ID < 12 )); then
  INDEX=$((TASK_ID - 6))
  DAMPING="${DAMPINGS[$INDEX]}"
  TAG="${DAMPING/./p}"
  python experiments/run_dfa_stall_comparison.py \
    --output-dir "results/dfa_stall_threefactor_dev_v1/error_d${TAG}" \
    --methods endfa --seeds 42 43 44 --feedback-seeds 0 \
    --damping 0.3 --error-damping "${DAMPING}" "${COMMON[@]}"
elif (( TASK_ID == 12 )); then
  python experiments/run_dfa_stall_comparison.py \
    --output-dir results/dfa_stall_threefactor_dev_v1/selected_joint \
    --methods dfa kndfa --seeds 42 43 44 --feedback-seeds 0 \
    --damping 0.3 --error-damping 10 "${COMMON[@]}"
else
  FEEDBACK_SEED=$((TASK_ID - 13))
  python experiments/run_dfa_stall_comparison.py \
    --output-dir "results/dfa_stall_threefactor_confirmation_v1/fb${FEEDBACK_SEED}" \
    --methods dfa ndfa endfa kndfa --seeds 50 51 52 53 54 \
    --feedback-seeds "${FEEDBACK_SEED}" \
    --damping 0.3 --error-damping 10 "${COMMON[@]}"
fi

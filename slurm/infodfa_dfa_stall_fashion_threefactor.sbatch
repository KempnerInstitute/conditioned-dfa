#!/usr/bin/env bash
# Pre-registered clean Fashion-MNIST replication of activity/error/K-nDFA.
# Tasks 0--7: activity-damping development grid.
# Tasks 8--15: error-damping development grid.
# Task 16: development-set joint/source-swap diagnostic at selected factors.
# Tasks 17--19: frozen confirmation, one task per feedback seed.
#SBATCH --job-name=infodfa_fashion_3factor
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-19%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
DAMPINGS=(0.03 0.1 0.3 1 3 10 30 100)
COMMON=(
  --dataset fashion_mnist
  --total-steps 1000
  --batch-size 128
  --hidden 300
  --lr 1e-3
  --norm-match-hidden
  --device cuda
  --validation-size 5000
  --validation-seed 24680
  --probe-n 5000
  --feature-probe-n 256
  --eval-every 50
  --metric-every 10
  --no-plots
)

if (( TASK_ID < 8 )); then
  DAMPING="${DAMPINGS[$TASK_ID]}"
  TAG="${DAMPING/./p}"
  python experiments/run_dfa_stall_comparison.py \
    --output-dir "results/dfa_stall_fashion_threefactor_dev_v1/activity_d${TAG}" \
    --methods ndfa --seeds 60 61 62 --feedback-seeds 0 \
    --damping "${DAMPING}" "${COMMON[@]}"
elif (( TASK_ID < 16 )); then
  INDEX=$((TASK_ID - 8))
  DAMPING="${DAMPINGS[$INDEX]}"
  TAG="${DAMPING/./p}"
  python experiments/run_dfa_stall_comparison.py \
    --output-dir "results/dfa_stall_fashion_threefactor_dev_v1/error_d${TAG}" \
    --methods endfa --seeds 60 61 62 --feedback-seeds 0 \
    --damping 0.3 --error-damping "${DAMPING}" "${COMMON[@]}"
elif (( TASK_ID == 16 )); then
  # The values below are placeholders until the two independent development
  # sweeps are analyzed; freeze them before launching tasks 16--19.
  : "${ACTIVITY_DAMPING:?set ACTIVITY_DAMPING from the activity-only validation sweep}"
  : "${ERROR_DAMPING:?set ERROR_DAMPING from the error-only validation sweep}"
  python experiments/run_dfa_stall_comparison.py \
    --output-dir results/dfa_stall_fashion_threefactor_dev_v1/selected_joint \
    --methods dfa kndfa kndfa_bp --seeds 60 61 62 --feedback-seeds 0 \
    --damping "${ACTIVITY_DAMPING}" --error-damping "${ERROR_DAMPING}" "${COMMON[@]}"
else
  : "${ACTIVITY_DAMPING:?set ACTIVITY_DAMPING from the activity-only validation sweep}"
  : "${ERROR_DAMPING:?set ERROR_DAMPING from the error-only validation sweep}"
  FEEDBACK_SEED=$((TASK_ID - 17))
  python experiments/run_dfa_stall_comparison.py \
    --output-dir "results/dfa_stall_fashion_threefactor_confirmation_v1/fb${FEEDBACK_SEED}" \
    --methods dfa ndfa endfa kndfa kndfa_bp --seeds 70 71 72 73 74 \
    --feedback-seeds "${FEEDBACK_SEED}" \
    --damping "${ACTIVITY_DAMPING}" --error-damping "${ERROR_DAMPING}" "${COMMON[@]}"
fi

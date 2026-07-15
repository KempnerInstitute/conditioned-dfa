#!/usr/bin/env bash
# P7 ReLU/softmax MNIST activity/error/K-nDFA replication.
# Tasks 0--6: activity-damping development grid.
# Tasks 7--14: error-damping development grid.
# Task 15: selected joint development check (no test evaluation).
# Tasks 16--18: frozen confirmation, one task per feedback seed.
#SBATCH --job-name=infodfa_relu3f
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-18%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
ACTIVITY_DAMPINGS=(0.03 0.1 0.3 1 3 10 30)
ERROR_DAMPINGS=(0.003 0.01 0.03 0.1 0.3 1 3 10)
COMMON=(
  --dataset mnist
  --hidden-dims 256 128
  --total-steps 1000
  --batch-size 128
  --lr 0.03
  --validation-size 5000
  --validation-seed 86420
  --probe-n 5000
  --eval-every 100
  --device cuda
)

if (( TASK_ID < 7 )); then
  DAMPING="${ACTIVITY_DAMPINGS[$TASK_ID]}"
  TAG="${DAMPING/./p}"
  python experiments/run_dfa_relu_vision_threefactor.py \
    --output-dir "results/dfa_relu_mnist_dev_v1/activity_d${TAG}" \
    --methods ndfa --seeds 0 1 2 --feedback-seeds 0 \
    --activity-damping "${DAMPING}" --error-damping 0.1 "${COMMON[@]}"
elif (( TASK_ID < 15 )); then
  INDEX=$((TASK_ID - 7))
  DAMPING="${ERROR_DAMPINGS[$INDEX]}"
  TAG="${DAMPING/./p}"
  python experiments/run_dfa_relu_vision_threefactor.py \
    --output-dir "results/dfa_relu_mnist_dev_v1/error_d${TAG}" \
    --methods endfa --seeds 0 1 2 --feedback-seeds 0 \
    --activity-damping 3 --error-damping "${DAMPING}" "${COMMON[@]}"
elif (( TASK_ID == 15 )); then
  python experiments/run_dfa_relu_vision_threefactor.py \
    --output-dir results/dfa_relu_mnist_dev_v1/selected_joint \
    --methods dfa ndfa endfa kndfa --seeds 0 1 2 --feedback-seeds 0 \
    --activity-damping 3 --error-damping 0.1 "${COMMON[@]}"
else
  FEEDBACK_SEED=$((TASK_ID - 16))
  python experiments/run_dfa_relu_vision_threefactor.py \
    --output-dir "results/dfa_relu_mnist_threefactor_confirmation_v1/fb${FEEDBACK_SEED}" \
    --methods dfa ndfa endfa kndfa \
    --seeds 100 101 102 103 104 105 106 107 \
    --feedback-seeds "${FEEDBACK_SEED}" \
    --activity-damping 3 --error-damping 0.1 --evaluate-test "${COMMON[@]}"
fi

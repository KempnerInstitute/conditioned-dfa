#!/usr/bin/env bash
# Post-hoc scale-matched audit of the Fashion-MNIST BP-error source comparator.
# Tasks 0--9: BP-source error-damping validation sweep.
# Tasks 10--12: fresh-seed confirmation, one task per feedback seed. Submit the
# latter with BP_ERROR_DAMPING set to the validation-selected value.
#SBATCH --job-name=infodfa_fashion_bpsource
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-9%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
DAMPINGS=(0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30)
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

if (( TASK_ID < 10 )); then
  DAMPING="${DAMPINGS[$TASK_ID]}"
  TAG="${DAMPING/./p}"
  python experiments/run_dfa_stall_comparison.py \
    --output-dir "results/dfa_stall_fashion_bpsource_retune_dev_v1/error_d${TAG}" \
    --methods kndfa_bp --seeds 60 61 62 --feedback-seeds 0 \
    --damping 0.03 --error-damping "${DAMPING}" --skip-test-eval \
    "${COMMON[@]}"
else
  : "${BP_ERROR_DAMPING:?Set BP_ERROR_DAMPING for confirmation tasks}"
  FEEDBACK_SEED=$((TASK_ID - 10))
  python experiments/run_dfa_stall_comparison.py \
    --output-dir "results/dfa_stall_fashion_bpsource_retune_confirmation_v1/local/fb${FEEDBACK_SEED}" \
    --methods ndfa kndfa --seeds 80 81 82 83 84 \
    --feedback-seeds "${FEEDBACK_SEED}" \
    --damping 0.03 --error-damping 30 "${COMMON[@]}"
  python experiments/run_dfa_stall_comparison.py \
    --output-dir "results/dfa_stall_fashion_bpsource_retune_confirmation_v1/bp/fb${FEEDBACK_SEED}" \
    --methods kndfa_bp --seeds 80 81 82 83 84 \
    --feedback-seeds "${FEEDBACK_SEED}" \
    --damping 0.03 --error-damping "${BP_ERROR_DAMPING}" "${COMMON[@]}"
fi

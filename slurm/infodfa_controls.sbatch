#!/usr/bin/env bash
#SBATCH --job-name=infodfa_controls
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm results

TASKS=(synthetic_nuisance fashion_mnist_noisy)
OUT_ROOTS=(results/dfa_controls_synthetic_nuisance_v1 results/dfa_controls_fashion_mnist_noisy_v1)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
OUT="${OUT_ROOTS[$SLURM_ARRAY_TASK_ID]}"

EXTRA_ARGS=""
if [ "$TASK" = "fashion_mnist_noisy" ]; then
  EXTRA_ARGS="--download"
fi

python experiments/run_dfa_controls.py \
  --task "$TASK" \
  --output-dir "$OUT" \
  --n-seeds 5 \
  --epochs 20 \
  --hidden-dims 256 128 \
  --batch-size 128 \
  --lr 0.08 \
  --label-noise 0.1 \
  --weight-decay 1e-3 \
  --label-smoothing 0.1 \
  --natural-damping 0.3 \
  --val-fraction 0.1 \
  --cuda \
  $EXTRA_ARGS

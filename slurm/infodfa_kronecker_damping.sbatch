#!/usr/bin/env bash
#SBATCH --job-name=dfa_kron
#SBATCH --partition=serial_requeue
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

DATASETS=(${DATASETS:-mnist fashion_mnist})
DAMPINGS=(${DAMPINGS:-0.001 0.003 0.01 0.03 0.1 0.3 1.0})
IDX="${SLURM_ARRAY_TASK_ID:-0}"
DATASET_IDX=$((IDX / ${#DAMPINGS[@]}))
DAMPING_IDX=$((IDX % ${#DAMPINGS[@]}))
DATASET="${DATASETS[$DATASET_IDX]}"
DAMPING="${DAMPINGS[$DAMPING_IDX]}"
SAFE_DAMPING="${DAMPING//./p}"
OUT_DIR="${OUT_ROOT:-results/dfa_kronecker_damping_v1}/${DATASET}/damping_${SAFE_DAMPING}"

python experiments/run_dfa_vision_baselines.py \
  --output-dir "$OUT_DIR" \
  --dataset "$DATASET" \
  --download \
  --methods bp dfa_random ndfa_random ndfa_random_kronecker \
  --n-seeds "${N_SEEDS:-3}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
  --n-train "${N_TRAIN:-10000}" \
  --n-test "${N_TEST:-2500}" \
  --hidden-dims 256 128 \
  --epochs "${EPOCHS:-8}" \
  --batch-size 128 \
  --lr 0.08 \
  --feedback-scale 1.0 \
  --feedback-rank "${FEEDBACK_RANK:-0}" \
  --natural-damping "$DAMPING" \
  --eval-size 512 \
  --local-pca-neighbors 24 \
  --local-pca-rank 8 \
  --seed "$IDX"

#!/usr/bin/env bash
#SBATCH --job-name=dfa_vision
#SBATCH --partition=serial_requeue
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

DATASETS=(${DATASETS:-mnist fashion_mnist cifar10})
IDX="${SLURM_ARRAY_TASK_ID:-0}"
DATASET="${DATASETS[$IDX]}"
OUT_DIR="${OUT_ROOT:-results/dfa_vision_full_slurm}/${DATASET}"

python experiments/run_dfa_vision_baselines.py \
  --output-dir "$OUT_DIR" \
  --dataset "$DATASET" \
  --download \
  --methods bp dfa_random dfa_bp_aligned_dynamic dfa_bp_orthogonal_dynamic ndfa_random ndfa_random_kronecker drtp_random \
  --n-seeds "${N_SEEDS:-3}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
  --n-train "${N_TRAIN:-12000}" \
  --n-test "${N_TEST:-3000}" \
  --hidden-dims 256 128 \
  --epochs "${EPOCHS:-8}" \
  --batch-size 128 \
  --lr 0.08 \
  --feedback-scale 1.0 \
  --feedback-rank "${FEEDBACK_RANK:-0}" \
  --natural-damping 1e-2 \
  --eval-size 512 \
  --local-pca-neighbors 24 \
  --local-pca-rank 8 \
  --seed "$IDX"

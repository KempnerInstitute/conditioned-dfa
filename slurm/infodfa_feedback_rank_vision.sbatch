#!/usr/bin/env bash
#SBATCH --job-name=dfa_rank_vis
#SBATCH --partition=serial_requeue
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm
export PYTHONUSERBASE="/n/home13/hsafaai/.local"

DATASETS=(${DATASETS:-mnist fashion_mnist})
RANKS=(${RANKS:-0 1 2 4 8 16})
IDX="${SLURM_ARRAY_TASK_ID:-0}"
DATASET_IDX=$((IDX / ${#RANKS[@]}))
RANK_IDX=$((IDX % ${#RANKS[@]}))
DATASET="${DATASETS[$DATASET_IDX]}"
RANK="${RANKS[$RANK_IDX]}"
OUT_DIR="${OUT_ROOT:-results/dfa_feedback_rank_vision_v1}/${DATASET}/rank_${RANK}"

python experiments/run_dfa_vision_baselines.py \
  --output-dir "$OUT_DIR" \
  --dataset "$DATASET" \
  --download \
  --methods bp dfa_random dfa_bp_aligned_dynamic dfa_bp_orthogonal_dynamic ndfa_random drtp_random \
  --n-seeds "${N_SEEDS:-3}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
  --n-train "${N_TRAIN:-10000}" \
  --n-test "${N_TEST:-2500}" \
  --hidden-dims 256 128 \
  --epochs "${EPOCHS:-8}" \
  --batch-size 128 \
  --lr 0.08 \
  --feedback-scale 1.0 \
  --feedback-rank "$RANK" \
  --natural-damping 1e-2 \
  --eval-size 512 \
  --local-pca-neighbors 24 \
  --local-pca-rank 8 \
  --seed "$IDX"

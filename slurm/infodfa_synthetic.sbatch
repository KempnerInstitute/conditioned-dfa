#!/usr/bin/env bash
#SBATCH --job-name=dfa_synth
#SBATCH --partition=serial_requeue
#SBATCH --time=08:00:00
#SBATCH --array=0-47%24
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"
N_SHARDS="${N_SHARDS:-48}"
OUT_ROOT="${OUT_ROOT:-results/dfa_synthetic_full_array}"
OUT_DIR="${OUT_ROOT}/shards/shard_${SHARD_INDEX}"
mkdir -p "${OUT_DIR}"

python experiments/run_dfa_synthetic.py \
  --output-dir "${OUT_DIR}" \
  --methods bp dfa_random dfa_bp_aligned_dynamic dfa_bp_orthogonal_dynamic dfa_tangent_biased dfa_tangent_orthogonal ndfa_random ndfa_random_kronecker drtp_random \
  --manifolds torus swiss_roll low_rank \
  --n-seeds "${N_SEEDS:-3}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
  --feedback-scales "${FEEDBACK_SCALE:-1.0}" \
  --feedback-ranks "${FEEDBACK_RANK:-0}" \
  --n-train "${N_TRAIN:-1536}" \
  --n-test "${N_TEST:-1536}" \
  --input-dim 10 \
  --input-noises ${INPUT_NOISES:-0.05 0.10} \
  --task-frequencies ${TASK_FREQUENCIES:-1 2} \
  --hidden-dims ${HIDDEN_DIMS:-64 128} \
  --n-hidden-layers-list "${DEPTH:-2}" \
  --epochs "${EPOCHS:-10}" \
  --early-epoch 3 \
  --batch-size 128 \
  --lr 0.25 \
  --eval-size 256 \
  --tangent-rank 2 \
  --natural-damping "${NATURAL_DAMPING:-0.01}" \
  --shard-index "${SHARD_INDEX}" \
  --n-shards "${N_SHARDS}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-4}" \
  --skip-analysis

#!/usr/bin/env bash
#SBATCH --job-name=dfa_rank_syn
#SBATCH --partition=serial_requeue
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm
export PYTHONUSERBASE="/n/home13/hsafaai/.local"

python experiments/run_dfa_synthetic.py \
  --output-dir "${OUT_DIR:-results/dfa_feedback_rank_synthetic_v1}" \
  --methods bp dfa_random dfa_tangent_biased dfa_tangent_orthogonal ndfa_random drtp_random \
  --manifolds torus swiss_roll low_rank \
  --n-seeds "${N_SEEDS:-3}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
  --feedback-scales "${FEEDBACK_SCALE:-1.0}" \
  --feedback-ranks ${FEEDBACK_RANKS:-0 1 2 4 8 16} \
  --n-train "${N_TRAIN:-1536}" \
  --n-test "${N_TEST:-1536}" \
  --input-dim 10 \
  --input-noises ${INPUT_NOISES:-0.05 0.10} \
  --task-frequencies ${TASK_FREQUENCIES:-1 2} \
  --hidden-dims ${HIDDEN_DIMS:-64 128} \
  --n-hidden-layers-list ${DEPTHS:-2} \
  --epochs "${EPOCHS:-10}" \
  --early-epoch 3 \
  --batch-size 128 \
  --lr 0.25 \
  --eval-size 256 \
  --tangent-rank 2 \
  --natural-damping "${NATURAL_DAMPING:-0.01}"

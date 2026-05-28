#!/usr/bin/env bash
#SBATCH --job-name=dfa_kndfa_spectrum
#SBATCH --partition=serial_requeue
#SBATCH --time=12:00:00
#SBATCH --array=0-47%16
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-Geo}"
mkdir -p logs/slurm

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"
N_SHARDS="${N_SHARDS:-48}"
OUT_ROOT="${OUT_ROOT:-results/dfa_preconditioning_spectrum_v1}"
OUT_DIR="${OUT_ROOT}/shards/shard_${SHARD_INDEX}"
mkdir -p "${OUT_DIR}"

python experiments/run_dfa_preconditioning_spectrum.py \
  --output-dir "${OUT_DIR}" \
  --conditions ${CONDITIONS:-task_aligned nuisance_dominant mixed_context low_sample_noisy} \
  --n-seeds "${N_SEEDS:-3}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
  --feedback-ranks ${FEEDBACK_RANKS:-0 2 8} \
  --gammas ${GAMMAS:-0.0 0.1 0.25 0.5 0.75 1.0} \
  --dampings ${DAMPINGS:-0.1 0.3 1.0} \
  --n-train "${N_TRAIN:-4096}" \
  --n-test "${N_TEST:-2048}" \
  --input-dim "${INPUT_DIM:-64}" \
  --n-classes "${N_CLASSES:-8}" \
  --nuisance-dim "${NUISANCE_DIM:-24}" \
  --input-noise "${INPUT_NOISE:-0.05}" \
  --hidden-dims ${HIDDEN_DIMS:-256 128} \
  --epochs "${EPOCHS:-12}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --lr "${LR:-0.08}" \
  --feedback-scale "${FEEDBACK_SCALE:-1.0}" \
  --eval-size "${EVAL_SIZE:-512}" \
  --shard-index "${SHARD_INDEX}" \
  --n-shards "${N_SHARDS}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-6}" \
  --skip-analysis

#!/usr/bin/env bash
#SBATCH --job-name=dfa_multiout
#SBATCH --partition=serial_requeue
#SBATCH --time=12:00:00
#SBATCH --array=0-63%16
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"
N_SHARDS="${N_SHARDS:-64}"
OUT_ROOT="${OUT_ROOT:-results/dfa_multioutput_synthetic_v1}"
OUT_DIR="${OUT_ROOT}/shards/shard_${SHARD_INDEX}"
mkdir -p "${OUT_DIR}"

python experiments/run_dfa_multioutput_synthetic.py \
  --output-dir "${OUT_DIR}" \
  --conditions ${CONDITIONS:-task_aligned nuisance_dominant mixed_context low_sample_noisy} \
  --methods ${METHODS:-bp dfa_random ndfa_random ndfa_random_kronecker drtp_random vnc nmnc} \
  --n-seeds "${N_SEEDS:-3}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
  --feedback-ranks ${FEEDBACK_RANKS:-0 1 2 4 8} \
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
  --natural-damping "${NATURAL_DAMPING:-0.3}" \
  --eval-size "${EVAL_SIZE:-512}" \
  --pca-size "${PCA_SIZE:-1024}" \
  --nc-update-intervals ${NC_UPDATE_INTERVALS:-10 100} \
  --nc-pca-update-interval "${NC_PCA_UPDATE_INTERVAL:-20}" \
  --nc-manifold-rank "${NC_MANIFOLD_RANK:-64}" \
  --nc-noise-scale "${NC_NOISE_SCALE:-0.05}" \
  --nc-feedback-lr "${NC_FEEDBACK_LR:-0.5}" \
  --nc-init "${NC_INIT:-zero}" \
  --shard-index "${SHARD_INDEX}" \
  --n-shards "${N_SHARDS}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-6}" \
  --skip-analysis

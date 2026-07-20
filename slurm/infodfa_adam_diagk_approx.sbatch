#!/usr/bin/env bash
# Adam/diagonal-K approximation tests for the Conditioned DFA paper.
#
# This job asks whether nDFA/K-nDFA gains can be explained by an Adam-like
# diagonal second-moment correction, or whether full activity/error covariance
# factors still add something beyond coordinatewise learning-rate changes.
# It writes shard CSVs consumed by slurm/infodfa_adam_diagk_approx_aggregate.sbatch.
#SBATCH --job-name=infodfa_adamk
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-63%16
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"
N_SHARDS="${N_SHARDS:-64}"
OUT_ROOT="${OUT_ROOT:-results/infodfa_adam_diagk_approx_v1}"
OUT_DIR="${OUT_ROOT}/shards/shard_${SHARD_INDEX}"
mkdir -p "${OUT_DIR}"

python experiments/run_infodfa_adam_diagk_approx.py \
  --output-dir "${OUT_DIR}" \
  --cells ${CELLS:-clean_aligned nuisance_hard mixed_hard low_sample_noisy} \
  --methods ${METHODS:-bp_sgd dfa_sgd dfa_adam_hidden dfa_diag_activity_sqrt dfa_diag_error_sqrt dfa_diag_k_sqrt ndfa_activity ndfa_error kndfa} \
  --n-seeds "${N_SEEDS:-5}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-2}" \
  --feedback-ranks ${FEEDBACK_RANKS:-0} \
  --dampings ${DAMPINGS:-0.03 0.1 0.3 1.0} \
  --adaptive-lrs ${ADAPTIVE_LRS:-0.001 0.003} \
  --n-test "${N_TEST:-4096}" \
  --input-dim "${INPUT_DIM:-64}" \
  --n-classes "${N_CLASSES:-8}" \
  --nuisance-dim "${NUISANCE_DIM:-24}" \
  --hidden-dims ${HIDDEN_DIMS:-256 128} \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --lr "${LR:-0.08}" \
  --feedback-scale "${FEEDBACK_SCALE:-1.0}" \
  --diagnostic-damping "${DIAGNOSTIC_DAMPING:-0.3}" \
  --eval-size "${EVAL_SIZE:-512}" \
  --shard-index "${SHARD_INDEX}" \
  --n-shards "${N_SHARDS}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-8}" \
  --cuda

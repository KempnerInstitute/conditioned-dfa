#!/usr/bin/env bash
# Norm-matched factor controls for the Conditioned DFA mechanism claim.
#
# This tests whether activity/error/K-nDFA gains survive after each local hidden
# gradient is rescaled to a matched BP layer norm. If nDFA+norm_match still beats
# DFA+norm_match, the gain cannot be explained by a scalar learning-rate or
# per-layer scale change.
#SBATCH --job-name=infodfa_normfac
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

CONDITIONS=(nuisance_dominant mixed_context low_sample_noisy task_aligned)
N_TRAINS=(512 1024 512 4096)
LABEL_NOISES=(0.2 0.2 0.4 0.0)
INPUT_NOISES=(0.15 0.15 0.15 0.05)
CELL_TAGS=(nuisance_hard mixed_hard low_sample_noisy clean_aligned)

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
if (( TASK_ID >= ${#CONDITIONS[@]} )); then
  echo "Task ${TASK_ID} >= ${#CONDITIONS[@]}; exiting."
  exit 0
fi

CONDITION="${CONDITIONS[$TASK_ID]}"
N_TRAIN_CELL="${N_TRAINS[$TASK_ID]}"
LABEL_NOISE="${LABEL_NOISES[$TASK_ID]}"
INPUT_NOISE="${INPUT_NOISES[$TASK_ID]}"
CELL_TAG="${CELL_TAGS[$TASK_ID]}"

OUT_ROOT="${OUT_ROOT:-results/infodfa_normmatch_factor_controls_v1}"
OUT_DIR="${OUT_ROOT}/${CELL_TAG}"
mkdir -p "${OUT_DIR}"

METHODS=(
  bp
  dfa_random
  dfa_random+norm_match
  ndfa_random
  ndfa_random+norm_match
  ndfa_random_error
  ndfa_random_error+norm_match
  ndfa_random_kronecker
  ndfa_random_kronecker+norm_match
)

python experiments/run_dfa_controls.py \
  --task synthetic_nuisance \
  --synthetic-condition "${CONDITION}" \
  --output-dir "${OUT_DIR}" \
  --methods "${METHODS[@]}" \
  --n-seeds "${N_SEEDS:-5}" \
  --n-train "${N_TRAIN_CELL}" \
  --n-test "${N_TEST:-4096}" \
  --input-dim "${INPUT_DIM:-64}" \
  --n-classes-synthetic "${N_CLASSES:-8}" \
  --nuisance-dim "${NUISANCE_DIM:-24}" \
  --input-noise "${INPUT_NOISE}" \
  --label-noise "${LABEL_NOISE}" \
  --hidden-dims ${HIDDEN_DIMS:-256 128} \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --lr "${LR:-0.08}" \
  --feedback-scale "${FEEDBACK_SCALE:-1.0}" \
  --feedback-rank "${FEEDBACK_RANK:-0}" \
  --natural-damping "${NATURAL_DAMPING:-0.3}" \
  --val-fraction "${VAL_FRACTION:-0.1}" \
  --cuda

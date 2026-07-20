#!/usr/bin/env bash
#SBATCH --job-name=infodfa_noise_synth
#SBATCH --partition=serial_requeue
#SBATCH --time=10:00:00
#SBATCH --array=0-127%32
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

CONDITIONS=(${CONDITIONS:-nuisance_dominant mixed_context low_sample_noisy task_aligned})
N_TRAINS=(${N_TRAINS:-512 1024 2048 4096})
LABEL_NOISES=(${LABEL_NOISES:-0.0 0.1 0.2 0.4})
INPUT_NOISES=(${INPUT_NOISES:-0.05 0.15})

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
N_CONDITIONS=${#CONDITIONS[@]}
N_NTRAINS=${#N_TRAINS[@]}
N_LABEL=${#LABEL_NOISES[@]}
N_INPUT=${#INPUT_NOISES[@]}
N_TOTAL=$((N_CONDITIONS * N_NTRAINS * N_LABEL * N_INPUT))
if (( TASK_ID >= N_TOTAL )); then
  echo "Task ${TASK_ID} >= ${N_TOTAL}; exiting."
  exit 0
fi

COND_INDEX=$((TASK_ID / (N_NTRAINS * N_LABEL * N_INPUT)))
REM=$((TASK_ID % (N_NTRAINS * N_LABEL * N_INPUT)))
NTRAIN_INDEX=$((REM / (N_LABEL * N_INPUT)))
REM=$((REM % (N_LABEL * N_INPUT)))
LABEL_INDEX=$((REM / N_INPUT))
INPUT_INDEX=$((REM % N_INPUT))

CONDITION="${CONDITIONS[$COND_INDEX]}"
N_TRAIN_CELL="${N_TRAINS[$NTRAIN_INDEX]}"
LABEL_NOISE="${LABEL_NOISES[$LABEL_INDEX]}"
INPUT_NOISE="${INPUT_NOISES[$INPUT_INDEX]}"

label_tag="${LABEL_NOISE//./p}"
input_tag="${INPUT_NOISE//./p}"
OUT_ROOT="${OUT_ROOT:-results/infodfa_multioutput_noise_sweep_v1}"
OUT_DIR="${OUT_ROOT}/${CONDITION}/ntrain_${N_TRAIN_CELL}/label_${label_tag}/input_${input_tag}"
mkdir -p "${OUT_DIR}"

python experiments/run_dfa_multioutput_synthetic.py \
  --output-dir "${OUT_DIR}" \
  --conditions "${CONDITION}" \
  --methods ${METHODS:-bp dfa_random fa_random ndfa_random ndfa_random_kronecker drtp_random vnc nmnc} \
  --n-seeds "${N_SEEDS:-5}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
  --feedback-ranks ${FEEDBACK_RANKS:-0 1 2 4 8} \
  --n-train "${N_TRAIN_CELL}" \
  --n-test "${N_TEST:-4096}" \
  --input-dim "${INPUT_DIM:-64}" \
  --n-classes "${N_CLASSES:-8}" \
  --nuisance-dim "${NUISANCE_DIM:-24}" \
  --input-noise "${INPUT_NOISE}" \
  --label-noise "${LABEL_NOISE}" \
  --hidden-dims ${HIDDEN_DIMS:-256 128} \
  --epochs "${EPOCHS:-14}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --lr "${LR:-0.08}" \
  --feedback-scale "${FEEDBACK_SCALE:-1.0}" \
  --natural-damping "${NATURAL_DAMPING:-0.3}" \
  --eval-size "${EVAL_SIZE:-512}" \
  --pca-size "${PCA_SIZE:-1024}" \
  --nc-update-intervals ${NC_UPDATE_INTERVALS:-10} \
  --nc-pca-update-interval "${NC_PCA_UPDATE_INTERVAL:-20}" \
  --nc-manifold-rank "${NC_MANIFOLD_RANK:-64}" \
  --nc-noise-scale "${NC_NOISE_SCALE:-0.05}" \
  --nc-feedback-lr "${NC_FEEDBACK_LR:-0.5}" \
  --nc-init "${NC_INIT:-zero}" \
  --covariance-diagnostics \
  --checkpoint-every "${CHECKPOINT_EVERY:-8}" \
  --skip-analysis

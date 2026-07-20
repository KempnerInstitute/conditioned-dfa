#!/usr/bin/env bash
#SBATCH --job-name=dfa_nmnc
#SBATCH --partition=serial_requeue
#SBATCH --time=10:00:00
#SBATCH --array=0-62%21
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

DATASETS=(${DATASETS:-mnist fashion_mnist cifar10})
UPDATE_INTERVALS=(${UPDATE_INTERVALS:-1 5 10 20 100 200 500})
MANIFOLD_RANKS=(${MANIFOLD_RANKS:-16 32 64})

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
N_DATASETS=${#DATASETS[@]}
N_INTERVALS=${#UPDATE_INTERVALS[@]}
N_RANKS=${#MANIFOLD_RANKS[@]}
N_TOTAL=$((N_DATASETS * N_INTERVALS * N_RANKS))
if (( TASK_ID >= N_TOTAL )); then
  echo "Task ${TASK_ID} >= ${N_TOTAL}; exiting."
  exit 0
fi

DATASET_INDEX=$((TASK_ID / (N_INTERVALS * N_RANKS)))
REM=$((TASK_ID % (N_INTERVALS * N_RANKS)))
INTERVAL_INDEX=$((REM / N_RANKS))
RANK_INDEX=$((REM % N_RANKS))

DATASET="${DATASETS[$DATASET_INDEX]}"
UPDATE_INTERVAL="${UPDATE_INTERVALS[$INTERVAL_INDEX]}"
MANIFOLD_RANK="${MANIFOLD_RANKS[$RANK_INDEX]}"

OUT_ROOT="${OUT_ROOT:-results/dfa_nmnc_comparison_v1}"
OUT_DIR="${OUT_ROOT}/${DATASET}/b_${UPDATE_INTERVAL}/rank_${MANIFOLD_RANK}"
mkdir -p "${OUT_DIR}"

if [[ "${DATASET}" == "cifar100" ]]; then
  HIDDEN_DIMS_DEFAULT="2048 1024"
  N_TRAIN_DEFAULT=20000
  N_TEST_DEFAULT=5000
  EPOCHS_DEFAULT=20
  LR_DEFAULT=0.02
elif [[ "${DATASET}" == "cifar10" ]]; then
  HIDDEN_DIMS_DEFAULT="1024 512"
  N_TRAIN_DEFAULT=12000
  N_TEST_DEFAULT=3000
  EPOCHS_DEFAULT=12
  LR_DEFAULT=0.04
else
  HIDDEN_DIMS_DEFAULT="512 256"
  N_TRAIN_DEFAULT=10000
  N_TEST_DEFAULT=2000
  EPOCHS_DEFAULT=10
  LR_DEFAULT=0.08
fi

python experiments/run_dfa_nmnc_comparison.py \
  --output-dir "${OUT_DIR}" \
  --dataset "${DATASET}" \
  --download \
  --methods ${METHODS:-bp dfa_random ndfa_random ndfa_random_kronecker drtp_random vnc nmnc} \
  --n-seeds "${N_SEEDS:-3}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
  --n-train "${N_TRAIN:-${N_TRAIN_DEFAULT}}" \
  --n-test "${N_TEST:-${N_TEST_DEFAULT}}" \
  --hidden-dims ${HIDDEN_DIMS:-${HIDDEN_DIMS_DEFAULT}} \
  --epochs "${EPOCHS:-${EPOCHS_DEFAULT}}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --lr "${LR:-${LR_DEFAULT}}" \
  --feedback-scale "${FEEDBACK_SCALE:-1.0}" \
  --feedback-rank "${FEEDBACK_RANK:-0}" \
  --natural-damping "${NATURAL_DAMPING:-0.3}" \
  --eval-size "${EVAL_SIZE:-512}" \
  --pca-size "${PCA_SIZE:-1024}" \
  --local-pca-neighbors "${LOCAL_PCA_NEIGHBORS:-24}" \
  --local-pca-rank "${LOCAL_PCA_RANK:-8}" \
  --nc-update-interval "${UPDATE_INTERVAL}" \
  --nc-pca-update-interval "${NC_PCA_UPDATE_INTERVAL:-20}" \
  --nc-manifold-rank "${MANIFOLD_RANK}" \
  --nc-noise-scale "${NC_NOISE_SCALE:-0.05}" \
  --nc-feedback-lr "${NC_FEEDBACK_LR:-0.5}" \
  --nc-init "${NC_INIT:-zero}"

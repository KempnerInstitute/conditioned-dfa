#!/usr/bin/env bash
#SBATCH --job-name=infodfa_noise_mlp
#SBATCH --partition=serial_requeue
#SBATCH --time=10:00:00
#SBATCH --array=0-23%12
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

DATASETS=(${DATASETS:-fashion_mnist cifar10})
N_TRAINS=(${N_TRAINS:-1000 3000 10000})
LABEL_NOISES=(${LABEL_NOISES:-0.0 0.1 0.2 0.4})

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
N_DATASETS=${#DATASETS[@]}
N_NTRAINS=${#N_TRAINS[@]}
N_LABEL=${#LABEL_NOISES[@]}
N_TOTAL=$((N_DATASETS * N_NTRAINS * N_LABEL))
if (( TASK_ID >= N_TOTAL )); then
  echo "Task ${TASK_ID} >= ${N_TOTAL}; exiting."
  exit 0
fi

DATASET_INDEX=$((TASK_ID / (N_NTRAINS * N_LABEL)))
REM=$((TASK_ID % (N_NTRAINS * N_LABEL)))
NTRAIN_INDEX=$((REM / N_LABEL))
LABEL_INDEX=$((REM % N_LABEL))

DATASET="${DATASETS[$DATASET_INDEX]}"
N_TRAIN_CELL="${N_TRAINS[$NTRAIN_INDEX]}"
LABEL_NOISE="${LABEL_NOISES[$LABEL_INDEX]}"
label_tag="${LABEL_NOISE//./p}"

if [[ "${DATASET}" == "cifar10" ]]; then
  HIDDEN_DIMS_DEFAULT="1024 512"
  N_TEST_DEFAULT=3000
  EPOCHS_DEFAULT=14
  LR_DEFAULT=0.04
else
  HIDDEN_DIMS_DEFAULT="512 256"
  N_TEST_DEFAULT=2000
  EPOCHS_DEFAULT=12
  LR_DEFAULT=0.08
fi

OUT_ROOT="${OUT_ROOT:-results/infodfa_vision_noise_sweep_v1}"
OUT_DIR="${OUT_ROOT}/${DATASET}/ntrain_${N_TRAIN_CELL}/label_${label_tag}"
mkdir -p "${OUT_DIR}"

python experiments/run_dfa_nmnc_comparison.py \
  --output-dir "${OUT_DIR}" \
  --dataset "${DATASET}" \
  --download \
  --methods ${METHODS:-bp dfa_random fa_random ndfa_random ndfa_random_kronecker drtp_random vnc nmnc} \
  --n-seeds "${N_SEEDS:-5}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
  --n-train "${N_TRAIN_CELL}" \
  --n-test "${N_TEST:-${N_TEST_DEFAULT}}" \
  --label-noise "${LABEL_NOISE}" \
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
  --nc-update-interval "${NC_UPDATE_INTERVAL:-10}" \
  --nc-pca-update-interval "${NC_PCA_UPDATE_INTERVAL:-20}" \
  --nc-manifold-rank "${NC_MANIFOLD_RANK:-64}" \
  --nc-noise-scale "${NC_NOISE_SCALE:-0.05}" \
  --nc-feedback-lr "${NC_FEEDBACK_LR:-0.5}" \
  --nc-init "${NC_INIT:-zero}" \
  --covariance-diagnostics

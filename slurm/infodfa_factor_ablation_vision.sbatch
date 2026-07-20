#!/usr/bin/env bash
# Focused activity/error/K-nDFA factor ablation on first-paper noisy vision MLP
# cells, with longer training to reveal saturation dynamics.
#SBATCH --job-name=infodfa_fact_vis
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

DATASETS=(fashion_mnist fashion_mnist cifar10 cifar10)
N_TRAINS=(1000 3000 1000 3000)
LABEL_NOISES=(0.4 0.4 0.4 0.4)
CELL_TAGS=(fashion_1k_ln40 fashion_3k_ln40 cifar10_1k_ln40 cifar10_3k_ln40)

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
if (( TASK_ID >= ${#DATASETS[@]} )); then
  echo "Task ${TASK_ID} >= ${#DATASETS[@]}; exiting."
  exit 0
fi

DATASET="${DATASETS[$TASK_ID]}"
N_TRAIN_CELL="${N_TRAINS[$TASK_ID]}"
LABEL_NOISE="${LABEL_NOISES[$TASK_ID]}"
CELL_TAG="${CELL_TAGS[$TASK_ID]}"

if [[ "${DATASET}" == "cifar10" ]]; then
  HIDDEN_DIMS_DEFAULT="1024 512"
  N_TEST_DEFAULT=3000
  LR_DEFAULT=0.04
else
  HIDDEN_DIMS_DEFAULT="512 256"
  N_TEST_DEFAULT=2000
  LR_DEFAULT=0.08
fi

OUT_ROOT="${OUT_ROOT:-results/infodfa_factor_ablation_vision_v1}"
OUT_DIR="${OUT_ROOT}/${CELL_TAG}"
mkdir -p "${OUT_DIR}"

python experiments/run_dfa_nmnc_comparison.py \
  --output-dir "${OUT_DIR}" \
  --dataset "${DATASET}" \
  --download \
  --methods ${METHODS:-bp dfa_random ndfa_random ndfa_random_error ndfa_random_kronecker} \
  --n-seeds "${N_SEEDS:-5}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
  --n-train "${N_TRAIN_CELL}" \
  --n-test "${N_TEST:-${N_TEST_DEFAULT}}" \
  --label-noise "${LABEL_NOISE}" \
  --hidden-dims ${HIDDEN_DIMS:-${HIDDEN_DIMS_DEFAULT}} \
  --epochs "${EPOCHS:-100}" \
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
  --covariance-diagnostics \
  --cuda

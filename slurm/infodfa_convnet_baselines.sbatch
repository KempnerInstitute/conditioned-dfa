#!/usr/bin/env bash
#SBATCH --job-name=dfa_convnet
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --array=0-17%6
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

DATASETS=(${DATASETS:-cifar10 cifar100})
RANKS=(${RANKS:-0 8 32})
DAMPINGS=(${DAMPINGS:-0.1 0.3 1.0})

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
N_DATASETS=${#DATASETS[@]}
N_RANKS=${#RANKS[@]}
N_DAMPINGS=${#DAMPINGS[@]}
N_TOTAL=$((N_DATASETS * N_RANKS * N_DAMPINGS))
if (( TASK_ID >= N_TOTAL )); then
  echo "Task ${TASK_ID} >= ${N_TOTAL}; exiting."
  exit 0
fi

DATASET_INDEX=$((TASK_ID / (N_RANKS * N_DAMPINGS)))
REM=$((TASK_ID % (N_RANKS * N_DAMPINGS)))
RANK_INDEX=$((REM / N_DAMPINGS))
DAMPING_INDEX=$((REM % N_DAMPINGS))

DATASET="${DATASETS[$DATASET_INDEX]}"
RANK="${RANKS[$RANK_INDEX]}"
DAMPING="${DAMPINGS[$DAMPING_INDEX]}"

METHOD_LIST=(${METHODS:-ndfa_random ndfa_random_kronecker})
if [[ "${DAMPING}" == "0.3" ]]; then
  METHOD_LIST=(dfa_random "${METHOD_LIST[@]}")
fi
if [[ "${DAMPING}" == "0.3" && "${RANK}" == "0" ]]; then
  METHOD_LIST=(bp "${METHOD_LIST[@]}")
fi

OUT_ROOT="${OUT_ROOT:-results/dfa_convnet_baselines_v1}"
OUT_DIR="${OUT_ROOT}/${DATASET}/rank_${RANK}/damping_${DAMPING//./p}"
mkdir -p "${OUT_DIR}"

if [[ "${DATASET}" == "cifar100" ]]; then
  N_CLASSES=100
  EPOCHS_DEFAULT=25
  N_TRAIN_DEFAULT=30000
  N_TEST_DEFAULT=5000
else
  N_CLASSES=10
  EPOCHS_DEFAULT=25
  N_TRAIN_DEFAULT=30000
  N_TEST_DEFAULT=5000
fi

python experiments/run_dfa_convnet_baselines.py \
  --output-dir "${OUT_DIR}" \
  --dataset "${DATASET}" \
  --download \
  --methods "${METHOD_LIST[@]}" \
  --n-seeds "${N_SEEDS:-2}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-2}" \
  --n-train "${N_TRAIN:-${N_TRAIN_DEFAULT}}" \
  --n-test "${N_TEST:-${N_TEST_DEFAULT}}" \
  --channels ${CHANNELS:-32 64 128} \
  --strides ${STRIDES:-1 2 2} \
  --epochs "${EPOCHS:-${EPOCHS_DEFAULT}}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --lr "${LR:-0.03}" \
  --feedback-scale "${FEEDBACK_SCALE:-1.0}" \
  --feedback-rank "${RANK}" \
  --natural-damping "${DAMPING}" \
  --eval-size "${EVAL_SIZE:-512}" \
  --n-classes "${N_CLASSES}" \
  --cuda \
  --skip-analysis

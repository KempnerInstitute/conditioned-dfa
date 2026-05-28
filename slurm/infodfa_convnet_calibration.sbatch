#!/usr/bin/env bash
#SBATCH --job-name=dfa_convcal
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-35%8
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-Geo}"
mkdir -p logs/slurm

MODES=(${MODES:-flat channel})
NORMS=(${NORMS:-none sqrt_target sqrt_flat})
LRS=(${LRS:-0.0001 0.0003 0.001})
SCALES=(${SCALES:-0.3 1.0})

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
N_MODES=${#MODES[@]}
N_NORMS=${#NORMS[@]}
N_LRS=${#LRS[@]}
N_SCALES=${#SCALES[@]}
N_TOTAL=$((N_MODES * N_NORMS * N_LRS * N_SCALES))
if (( TASK_ID >= N_TOTAL )); then
  echo "Task ${TASK_ID} >= ${N_TOTAL}; exiting."
  exit 0
fi

MODE_INDEX=$((TASK_ID / (N_NORMS * N_LRS * N_SCALES)))
REM=$((TASK_ID % (N_NORMS * N_LRS * N_SCALES)))
NORM_INDEX=$((REM / (N_LRS * N_SCALES)))
REM=$((REM % (N_LRS * N_SCALES)))
LR_INDEX=$((REM / N_SCALES))
SCALE_INDEX=$((REM % N_SCALES))

MODE="${MODES[$MODE_INDEX]}"
NORM="${NORMS[$NORM_INDEX]}"
LR="${LRS[$LR_INDEX]}"
SCALE="${SCALES[$SCALE_INDEX]}"

METHOD_LIST=(${METHODS:-dfa_random ndfa_random ndfa_random_kronecker})
if [[ "${MODE}" == "channel" && "${NORM}" == "sqrt_flat" && "${LR}" == "0.001" && "${SCALE}" == "1.0" ]]; then
  METHOD_LIST=(bp "${METHOD_LIST[@]}")
fi

OUT_ROOT="${OUT_ROOT:-results/dfa_convnet_calibration_v1}"
LR_TAG="${LR//./p}"
SCALE_TAG="${SCALE//./p}"
OUT_DIR="${OUT_ROOT}/cifar10/mode_${MODE}/norm_${NORM}/lr_${LR_TAG}/scale_${SCALE_TAG}"
mkdir -p "${OUT_DIR}"

python experiments/run_dfa_convnet_baselines.py \
  --output-dir "${OUT_DIR}" \
  --dataset cifar10 \
  --download \
  --methods "${METHOD_LIST[@]}" \
  --n-seeds "${N_SEEDS:-2}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-2}" \
  --n-train "${N_TRAIN:-30000}" \
  --n-test "${N_TEST:-5000}" \
  --channels ${CHANNELS:-32 64 128} \
  --strides ${STRIDES:-1 2 2} \
  --epochs "${EPOCHS:-35}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --lr "${LR}" \
  --optimizer "${OPTIMIZER:-adam}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --feedback-mode "${MODE}" \
  --feedback-normalization "${NORM}" \
  --feedback-scale "${SCALE}" \
  --feedback-rank "${FEEDBACK_RANK:-0}" \
  --natural-damping "${NATURAL_DAMPING:-0.3}" \
  --eval-size "${EVAL_SIZE:-512}" \
  --n-classes 10 \
  --cuda \
  --skip-analysis

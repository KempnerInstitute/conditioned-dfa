#!/usr/bin/env bash
#SBATCH --job-name=dfa_convscale
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --array=0-25%6
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-Geo}"
mkdir -p logs/slurm

DATASETS=(${DATASETS:-cifar10 cifar100})
MODES=(${MODES:-flat channel})
LRS=(${LRS:-0.0001 0.0003 0.001})
BP_LRS=(${BP_LRS:-${BP_LR:-0.001}})
SCALES=(${SCALES:-0.3 1.0})

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
N_DATASETS=${#DATASETS[@]}
N_MODES=${#MODES[@]}
N_LRS=${#LRS[@]}
N_BP_LRS=${#BP_LRS[@]}
N_SCALES=${#SCALES[@]}
N_LOCAL=$((N_DATASETS * N_MODES * N_LRS * N_SCALES))
N_BP=$((N_DATASETS * N_BP_LRS))
N_TOTAL=$((N_LOCAL + N_BP))
if (( TASK_ID >= N_TOTAL )); then
  echo "Task ${TASK_ID} >= ${N_TOTAL}; exiting."
  exit 0
fi

if (( TASK_ID < N_LOCAL )); then
  DATASET_INDEX=$((TASK_ID / (N_MODES * N_LRS * N_SCALES)))
  REM=$((TASK_ID % (N_MODES * N_LRS * N_SCALES)))
  MODE_INDEX=$((REM / (N_LRS * N_SCALES)))
  REM=$((REM % (N_LRS * N_SCALES)))
  LR_INDEX=$((REM / N_SCALES))
  SCALE_INDEX=$((REM % N_SCALES))
  DATASET="${DATASETS[$DATASET_INDEX]}"
  MODE="${MODES[$MODE_INDEX]}"
  LR="${LRS[$LR_INDEX]}"
  SCALE="${SCALES[$SCALE_INDEX]}"
  METHOD_LIST=(${METHODS:-dfa_random ndfa_random ndfa_random_kronecker})
  NORM="${NORM:-sqrt_flat}"
else
  BP_TASK=$((TASK_ID - N_LOCAL))
  DATASET_INDEX=$((BP_TASK / N_BP_LRS))
  BP_LR_INDEX=$((BP_TASK % N_BP_LRS))
  DATASET="${DATASETS[$DATASET_INDEX]}"
  MODE="none"
  LR="${BP_LRS[$BP_LR_INDEX]}"
  SCALE="1.0"
  NORM="none"
  METHOD_LIST=(bp)
fi

OUT_ROOT="${OUT_ROOT:-results/dfa_convnet_scaled_v1}"
LR_TAG="${LR//./p}"
SCALE_TAG="${SCALE//./p}"
OUT_DIR="${OUT_ROOT}/${DATASET}/mode_${MODE}/norm_${NORM}/lr_${LR_TAG}/scale_${SCALE_TAG}"
mkdir -p "${OUT_DIR}"

DOWNLOAD_ARGS=()
if [[ "${DOWNLOAD:-1}" == "1" ]]; then
  DOWNLOAD_ARGS=(--download)
fi
BALANCED_ARGS=()
if [[ "${BALANCED_SUBSAMPLE:-0}" == "1" ]]; then
  BALANCED_ARGS=(--balanced-subsample)
fi

python experiments/run_dfa_convnet_baselines.py \
  --output-dir "${OUT_DIR}" \
  --dataset "${DATASET}" \
  --data-dir "${DATA_DIR:-data/torchvision}" \
  --imagenet-root "${IMAGENET_ROOT:-}" \
  --imagenet-classes "${IMAGENET_CLASSES:-0}" \
  --class-subset-seed "${CLASS_SUBSET_SEED:-0}" \
  --image-size "${IMAGE_SIZE:-64}" \
  "${DOWNLOAD_ARGS[@]}" \
  "${BALANCED_ARGS[@]}" \
  --methods "${METHOD_LIST[@]}" \
  --n-seeds "${N_SEEDS:-2}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-2}" \
  --n-train "${N_TRAIN:-50000}" \
  --n-test "${N_TEST:-10000}" \
  --channels ${CHANNELS:-64 128 256} \
  --strides ${STRIDES:-2 2 2} \
  --kernels ${KERNELS:-5 5 3} \
  --paddings ${PADDINGS:-2 2 1} \
  --fc-hidden ${FC_HIDDEN:-1024} \
  --epochs "${EPOCHS:-80}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --lr "${LR}" \
  --optimizer "${OPTIMIZER:-adam}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --feedback-mode "${MODE/none/channel}" \
  --feedback-normalization "${NORM/none/sqrt_flat}" \
  --feedback-scale "${SCALE}" \
  --feedback-rank "${FEEDBACK_RANK:-0}" \
  --natural-damping "${NATURAL_DAMPING:-0.3}" \
  --eval-size "${EVAL_SIZE:-1024}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-256}" \
  --cuda \
  --skip-analysis

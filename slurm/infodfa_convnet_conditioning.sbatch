#!/usr/bin/env bash
#SBATCH --job-name=dfa_convcond
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --array=0-91%6
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-Geo}"
mkdir -p logs/slurm

DATASETS=(${DATASETS:-cifar10})
RANKS=(${RANKS:-1 2 4 8 0})
DAMPINGS=(${DAMPINGS:-0.03 0.1 0.3 1.0})

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"

SPECS=()
for DATASET in "${DATASETS[@]}"; do
  SPECS+=("${DATASET}:bp:0:0.3")
  for RANK in "${RANKS[@]}"; do
    SPECS+=("${DATASET}:dfa_random:${RANK}:0.3")
    for DAMPING in "${DAMPINGS[@]}"; do
      SPECS+=("${DATASET}:ndfa_random:${RANK}:${DAMPING}")
      SPECS+=("${DATASET}:ndfa_random_kronecker:${RANK}:${DAMPING}")
    done
  done
done

N_TOTAL=${#SPECS[@]}
if (( TASK_ID >= N_TOTAL )); then
  echo "Task ${TASK_ID} >= ${N_TOTAL}; exiting."
  exit 0
fi

IFS=: read -r DATASET METHOD RANK DAMPING <<< "${SPECS[$TASK_ID]}"
RANK_TAG=$([[ "${RANK}" == "0" ]] && echo "full" || echo "${RANK}")
DAMP_TAG="${DAMPING//./p}"
OUT_ROOT="${OUT_ROOT:-results/dfa_convnet_conditioning_v1}"
OUT_DIR="${OUT_ROOT}/${DATASET}/${METHOD}/rank_${RANK_TAG}/damping_${DAMP_TAG}"
mkdir -p "${OUT_DIR}"

python experiments/run_dfa_convnet_baselines.py \
  --output-dir "${OUT_DIR}" \
  --dataset "${DATASET}" \
  --download \
  --methods "${METHOD}" \
  --n-seeds "${N_SEEDS:-1}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-2}" \
  --n-train "${N_TRAIN:-50000}" \
  --n-test "${N_TEST:-10000}" \
  --channels ${CHANNELS:-64 128 256} \
  --strides ${STRIDES:-2 2 2} \
  --kernels ${KERNELS:-5 5 3} \
  --paddings ${PADDINGS:-2 2 1} \
  --fc-hidden ${FC_HIDDEN:-1024} \
  --epochs "${EPOCHS:-60}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --lr "${LR:-0.0003}" \
  --optimizer "${OPTIMIZER:-adam}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --gradient-clip "${GRADIENT_CLIP:-0.0}" \
  --feedback-mode "${FEEDBACK_MODE:-channel}" \
  --feedback-normalization "${FEEDBACK_NORMALIZATION:-sqrt_flat}" \
  --feedback-scale "${FEEDBACK_SCALE:-0.3}" \
  --feedback-rank "${RANK}" \
  --natural-damping "${DAMPING}" \
  --eval-size "${EVAL_SIZE:-1024}" \
  --cuda \
  --skip-analysis

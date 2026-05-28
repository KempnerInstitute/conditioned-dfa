#!/usr/bin/env bash
#SBATCH --job-name=infodfa_c100_hard
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-Geo}"
mkdir -p logs/slurm

METHODS=(bp local_loss dfa_random ndfa_random ndfa_random_kronecker drtp_random)
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
if (( TASK_ID >= ${#METHODS[@]} )); then
  echo "Task ${TASK_ID} >= ${#METHODS[@]}; exiting."
  exit 0
fi

METHOD="${METHODS[$TASK_ID]}"
OUT_ROOT="${OUT_ROOT:-results/infodfa_hard_cifar100_confirm_v1}"
OUT_DIR="${OUT_ROOT}/${METHOD}"
mkdir -p "${OUT_DIR}"

case "${METHOD}" in
  bp)
    LR_DEFAULT="${BP_LR:-0.0003}"
    FEEDBACK_MODE_DEFAULT="channel"
    FEEDBACK_NORM_DEFAULT="sqrt_flat"
    ;;
  local_loss)
    LR_DEFAULT="${LOCAL_LR:-0.0001}"
    FEEDBACK_MODE_DEFAULT="channel"
    FEEDBACK_NORM_DEFAULT="sqrt_flat"
    ;;
  ndfa_random|ndfa_random_kronecker)
    LR_DEFAULT="${NDFA_LR:-0.0001}"
    FEEDBACK_MODE_DEFAULT="channel"
    FEEDBACK_NORM_DEFAULT="sqrt_flat"
    ;;
  *)
    LR_DEFAULT="${DFA_LR:-0.0003}"
    FEEDBACK_MODE_DEFAULT="channel"
    FEEDBACK_NORM_DEFAULT="sqrt_flat"
    ;;
esac

python experiments/run_dfa_convnet_baselines.py \
  --output-dir "${OUT_DIR}" \
  --dataset cifar100 \
  --download \
  --methods "${METHOD}" \
  --n-seeds "${N_SEEDS:-5}" \
  --n-feedback-seeds "${N_FEEDBACK_SEEDS:-5}" \
  --n-train "${N_TRAIN:-50000}" \
  --n-test "${N_TEST:-10000}" \
  --channels ${CHANNELS:-64 128 256} \
  --strides ${STRIDES:-1 2 2} \
  --fc-hidden ${FC_HIDDEN:-512} \
  --epochs "${EPOCHS:-40}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --optimizer "${OPTIMIZER:-adam}" \
  --lr "${LR:-${LR_DEFAULT}}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --gradient-clip "${GRADIENT_CLIP:-0.0}" \
  --feedback-scale "${FEEDBACK_SCALE:-0.3}" \
  --feedback-mode "${FEEDBACK_MODE:-${FEEDBACK_MODE_DEFAULT}}" \
  --feedback-normalization "${FEEDBACK_NORMALIZATION:-${FEEDBACK_NORM_DEFAULT}}" \
  --feedback-rank "${FEEDBACK_RANK:-0}" \
  --natural-damping "${NATURAL_DAMPING:-0.3}" \
  --eval-size "${EVAL_SIZE:-1024}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-512}" \
  --cuda \
  --skip-analysis

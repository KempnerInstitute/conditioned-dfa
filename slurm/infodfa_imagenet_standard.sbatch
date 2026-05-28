#!/usr/bin/env bash
#SBATCH --job-name=imgnet_ca
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --array=0-3%2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-Geo}"
mkdir -p logs/slurm

ARCHS=(${ARCHS:-resnet18})
METHODS=(${METHODS:-bp block_dfa block_ndfa local_aux})
MODULE_SETS=(${FEEDBACK_MODULE_SETS:-${FEEDBACK_MODULES:-auto}})
SPATIAL_MODES=(${SPATIAL_MODES:-broadcast})
LRS=(${LRS:-0.2})
SCALES=(${SCALES:-1.0})
GAMMAS=(${GAMMAS:-1.0})
MODULE_SCALE_SCHEDULES=(${MODULE_SCALE_SCHEDULES:-none})
DAMPINGS=(${DAMPINGS:-0.1})
SEEDS=(${SEEDS:-0})
FEEDBACK_SEEDS=(${FEEDBACK_SEEDS:-0})

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
N_ARCH=${#ARCHS[@]}
N_METHOD=${#METHODS[@]}
N_MODULE=${#MODULE_SETS[@]}
N_SPATIAL=${#SPATIAL_MODES[@]}
N_LR=${#LRS[@]}
N_SCALE=${#SCALES[@]}
N_GAMMA=${#GAMMAS[@]}
N_MODULE_SCALE=${#MODULE_SCALE_SCHEDULES[@]}
N_DAMP=${#DAMPINGS[@]}
N_SEED=${#SEEDS[@]}
N_FB=${#FEEDBACK_SEEDS[@]}
N_TOTAL=$((N_ARCH * N_METHOD * N_MODULE * N_SPATIAL * N_LR * N_SCALE * N_GAMMA * N_MODULE_SCALE * N_DAMP * N_SEED * N_FB))
if (( TASK_ID >= N_TOTAL )); then
  echo "Task ${TASK_ID} >= ${N_TOTAL}; exiting."
  exit 0
fi

REM=${TASK_ID}
FB_INDEX=$((REM % N_FB)); REM=$((REM / N_FB))
SEED_INDEX=$((REM % N_SEED)); REM=$((REM / N_SEED))
DAMP_INDEX=$((REM % N_DAMP)); REM=$((REM / N_DAMP))
MODULE_SCALE_INDEX=$((REM % N_MODULE_SCALE)); REM=$((REM / N_MODULE_SCALE))
GAMMA_INDEX=$((REM % N_GAMMA)); REM=$((REM / N_GAMMA))
SCALE_INDEX=$((REM % N_SCALE)); REM=$((REM / N_SCALE))
LR_INDEX=$((REM % N_LR)); REM=$((REM / N_LR))
SPATIAL_INDEX=$((REM % N_SPATIAL)); REM=$((REM / N_SPATIAL))
MODULE_INDEX=$((REM % N_MODULE)); REM=$((REM / N_MODULE))
METHOD_INDEX=$((REM % N_METHOD)); REM=$((REM / N_METHOD))
ARCH_INDEX=$((REM % N_ARCH))

ARCH="${ARCHS[$ARCH_INDEX]}"
METHOD="${METHODS[$METHOD_INDEX]}"
FEEDBACK_MODULE_SET="${MODULE_SETS[$MODULE_INDEX]}"
SPATIAL_MODE="${SPATIAL_MODES[$SPATIAL_INDEX]}"
LR="${LRS[$LR_INDEX]}"
SCALE="${SCALES[$SCALE_INDEX]}"
GAMMA="${GAMMAS[$GAMMA_INDEX]}"
MODULE_SCALE_SCHEDULE="${MODULE_SCALE_SCHEDULES[$MODULE_SCALE_INDEX]}"
DAMPING="${DAMPINGS[$DAMP_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
FEEDBACK_SEED="${FEEDBACK_SEEDS[$FB_INDEX]}"

LR_TAG="${LR//./p}"
SCALE_TAG="${SCALE//./p}"
GAMMA_TAG="${GAMMA//./p}"
DAMP_TAG="${DAMPING//./p}"
MODULE_TAG="${FEEDBACK_MODULE_SET//,/+}"
MODULE_TAG="${MODULE_TAG//./_}"
SPATIAL_TAG="${SPATIAL_MODE//./_}"
MODULE_SCALE_TAG="${MODULE_SCALE_SCHEDULE//,/+}"
MODULE_SCALE_TAG="${MODULE_SCALE_TAG//:/_}"
MODULE_SCALE_TAG="${MODULE_SCALE_TAG//./p}"
OUT_ROOT="${OUT_ROOT:-results/imagenet_credit_assignment_standard_v1}"
OUT_DIR="${OUT_ROOT}/${ARCH}/${METHOD}/modules_${MODULE_TAG}/spatial_${SPATIAL_TAG}/lr_${LR_TAG}/scale_${SCALE_TAG}/gamma_${GAMMA_TAG}/module_scales_${MODULE_SCALE_TAG}/damping_${DAMP_TAG}/seed_${SEED}/fb_${FEEDBACK_SEED}"
mkdir -p "${OUT_DIR}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
AMP_ARGS=()
if [[ "${AMP:-1}" == "1" ]]; then AMP_ARGS=(--amp); fi
CHANNEL_ARGS=()
if [[ "${CHANNELS_LAST:-1}" == "1" ]]; then CHANNEL_ARGS=(--channels-last); fi
COMPILE_ARGS=()
if [[ "${COMPILE_MODEL:-0}" == "1" ]]; then COMPILE_ARGS=(--compile); fi
RANDAUG_ARGS=()
if [[ "${RANDAUGMENT:-0}" == "1" ]]; then RANDAUG_ARGS=(--randaugment); fi
EVAL_EMA_ARGS=()
if [[ "${EVAL_EMA:-0}" == "1" ]]; then EVAL_EMA_ARGS=(--eval-ema); fi
STOP_NONFINITE_ARGS=()
if [[ "${STOP_ON_NONFINITE:-0}" == "1" ]]; then STOP_NONFINITE_ARGS=(--stop-on-nonfinite); fi
MODULE_SCALE_ARGS=()
if [[ "${MODULE_SCALE_SCHEDULE}" != "none" ]]; then MODULE_SCALE_ARGS=(--feedback-module-scales "${MODULE_SCALE_SCHEDULE}"); fi

python -m torch.distributed.run --standalone --nproc_per_node="${GPUS_PER_NODE}" experiments/run_imagenet_credit_assignment.py \
  --dataset "${DATASET:-imagenet1k}" \
  --imagenet-root "${IMAGENET_ROOT:-/n/holystore01/LABS/barak_lab/Everyone/datasets/imagenet}" \
  --output-dir "${OUT_DIR}" \
  --arch "${ARCH}" \
  --model-source "${MODEL_SOURCE:-auto}" \
  --method "${METHOD}" \
  --feedback-modules "${FEEDBACK_MODULE_SET}" \
  --feedback-seed "${FEEDBACK_SEED}" \
  --feedback-scale "${SCALE}" \
  --feedback-blend-gamma "${GAMMA}" \
  --feedback-spatial-mode "${SPATIAL_MODE}" \
  --dfa-norm "${DFA_NORM:-bp}" \
  --natural-damping "${DAMPING}" \
  --aux-weight "${AUX_WEIGHT:-0.3}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-0.0}" \
  --epochs "${EPOCHS:-90}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --workers "${WORKERS:-8}" \
  --optimizer "${OPTIMIZER:-sgd}" \
  --lr "${LR}" \
  --momentum "${MOMENTUM:-0.9}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --label-smoothing "${LABEL_SMOOTHING:-0.0}" \
  --random-erasing "${RANDOM_ERASING:-0.0}" \
  --mixup-alpha "${MIXUP_ALPHA:-0.0}" \
  --cutmix-alpha "${CUTMIX_ALPHA:-0.0}" \
  --mixup-prob "${MIXUP_PROB:-1.0}" \
  --ema-decay "${EMA_DECAY:-0.0}" \
  --repeated-aug "${REPEATED_AUG:-1}" \
  --drop-path-rate "${DROP_PATH_RATE:-0.0}" \
  --warmup-epochs "${WARMUP_EPOCHS:-5}" \
  --image-size "${IMAGE_SIZE:-224}" \
  --val-resize "${VAL_RESIZE:-256}" \
  --subsample-train "${SUBSAMPLE_TRAIN:-0}" \
  --subsample-val "${SUBSAMPLE_VAL:-0}" \
  --label-noise "${LABEL_NOISE:-0.0}" \
  --class-subset "${CLASS_SUBSET:-0}" \
  --class-subset-seed "${CLASS_SUBSET_SEED:-0}" \
  --seed "${SEED}" \
  --amp-init-scale "${AMP_INIT_SCALE:-65536.0}" \
  --amp-growth-interval "${AMP_GROWTH_INTERVAL:-2000}" \
  --print-freq "${PRINT_FREQ:-200}" \
  "${AMP_ARGS[@]}" \
  "${CHANNEL_ARGS[@]}" \
  "${RANDAUG_ARGS[@]}" \
  "${EVAL_EMA_ARGS[@]}" \
  "${STOP_NONFINITE_ARGS[@]}" \
  "${MODULE_SCALE_ARGS[@]}" \
  "${COMPILE_ARGS[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=infodfa_in100_seeds
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-25%4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Multi-seed extension of the strong-form ImageNet-100 substitution-depth sweep
# (reviewer ask: error bars on the ~20pp all-block cliff and per-depth orderings).
# Seeds 1 and 2 for all 13 configs; seed 0 is the existing strongform_v1 run.
# Per-config LR matches the headline table: full-covariance at layer2+3+4 and
# all-block uses eta=0.01 (the lr-check showed eta=0.1 collapses there).
set -euo pipefail
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate /n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2
cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

export IMAGENET_ROOT="${IMAGENET_ROOT:-/n/holystore01/LABS/barak_lab/Everyone/datasets/imagenet}"

METHODS=( bp block_dfa block_dfa block_dfa block_dfa block_ndfa block_ndfa block_ndfa block_ndfa block_ndfa block_ndfa block_ndfa block_ndfa)
MODULES=( none layer4 "layer3,layer4" "layer2,layer3,layer4" "layer1,layer2,layer3,layer4" \
          layer4 "layer3,layer4" "layer2,layer3,layer4" "layer1,layer2,layer3,layer4" \
          layer4 "layer3,layer4" "layer2,layer3,layer4" "layer1,layer2,layer3,layer4")
WHITEN=(  diag diag diag diag diag diag diag diag diag full full full full)
TAGS=(    bp dfa_layer4 dfa_l34 dfa_l234 dfa_all ndfaDiag_layer4 ndfaDiag_l34 ndfaDiag_l234 ndfaDiag_all ndfaFull_layer4 ndfaFull_l34 ndfaFull_l234 ndfaFull_all)
LRS=(     0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.01 0.01)

i="${SLURM_ARRAY_TASK_ID:-0}"
CFG=$(( i % 13 ))
SEED=$(( i / 13 + 1 ))   # seeds 1 and 2
METHOD="${METHODS[$CFG]}"; MODSET="${MODULES[$CFG]}"; WMODE="${WHITEN[$CFG]}"
TAG="${TAGS[$CFG]}"; LR="${LRS[$CFG]}"
OUT_DIR="${OUT_ROOT:-results/imagenet100_strongform_v1}/${TAG}_seed${SEED}"
mkdir -p "${OUT_DIR}"

COMMON=(--dataset imagenet1k --imagenet-root "${IMAGENET_ROOT}" --output-dir "${OUT_DIR}"
        --arch resnet18 --model-source torchvision --pretrained
        --epochs "${EPOCHS:-90}" --batch-size "${BATCH_SIZE:-256}" --workers 16
        --optimizer sgd --lr "${LR}" --momentum 0.9 --weight-decay 1e-4
        --warmup-epochs 5 --image-size 176 --val-resize 232
        --class-subset 100 --class-subset-seed 0 --seed "${SEED}")

run_once() {
  if [ "$METHOD" = "bp" ]; then
    python -m torch.distributed.run --standalone --nproc_per_node=1 \
      experiments/run_imagenet_credit_assignment.py --method bp "${COMMON[@]}"
  else
    python -m torch.distributed.run --standalone --nproc_per_node=1 \
      experiments/run_imagenet_credit_assignment.py --method "$METHOD" \
      --feedback-modules "$MODSET" --feedback-seed "${SEED}" --feedback-spatial-mode broadcast \
      --dfa-norm unit --whiten-mode "$WMODE" --natural-damping 0.1 "${COMMON[@]}"
  fi
}
for attempt in 1 2 3; do
  if run_once; then echo "succeeded on attempt ${attempt}"; break; fi
  echo "attempt ${attempt} failed (likely transient CUDA); sleeping before retry"
  sleep 90
done

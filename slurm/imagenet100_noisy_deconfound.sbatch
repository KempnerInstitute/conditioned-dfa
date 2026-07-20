#!/usr/bin/env bash
#SBATCH --job-name=infodfa_in100_noisy_deconf
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-20%8
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Noisy ImageNet-100 deconfound (scale x regime): rerun the strong-form
# substitution-depth protocol with 40% symmetric train-label noise so the
# noise-dominant regime is tested AT SCALE rather than only on MLPs. If the
# conditioning gain is regime-driven (not scale-driven), the noisy variant
# should recover/extend the nDFA advantage where the clean strongform table
# shows the depth cliff. Label noise is applied once at dataset construction,
# train set only, deterministic given --seed (see NoisyLabelDataset).
#
# Grid: {raw block-DFA, diagonal channel-nDFA, full-cov ZCA} x depths
# {layer4, all-block} + BP (depth-free), x 3 seeds = 21 tasks; this covers the
# 4-method x 2-depth x 3-seed grid without duplicating the depth-independent
# BP runs. Everything else matches infodfa_imagenet_strongform_seeds.sbatch
# (90 epochs, lr 0.1 except full-cov all-block at 0.01, --dfa-norm unit,
# image 176, pretrained torchvision ResNet-18).
set -euo pipefail
# Optionally activate a conda env by exporting CONDA_ENV before submitting;
# otherwise the environment active at submission time is used.
if [ -n "${CONDA_ENV:-}" ]; then
    source "${CONDA_SH:-/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV}"
fi
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

export IMAGENET_ROOT="${IMAGENET_ROOT:?set IMAGENET_ROOT to an ImageNet-1k ImageFolder}"
LABEL_NOISE="${LABEL_NOISE:-0.4}"

METHODS=( bp block_dfa block_dfa block_ndfa block_ndfa block_ndfa block_ndfa )
MODULES=( none layer4 "layer1,layer2,layer3,layer4" \
          layer4 "layer1,layer2,layer3,layer4" \
          layer4 "layer1,layer2,layer3,layer4" )
WHITEN=(  diag diag diag diag diag full full )
TAGS=(    bp dfa_layer4 dfa_all ndfaDiag_layer4 ndfaDiag_all ndfaFull_layer4 ndfaFull_all )
LRS=(     0.1 0.1 0.1 0.1 0.1 0.1 0.01 )

i="${SLURM_ARRAY_TASK_ID:-0}"
CFG=$(( i % 7 ))
SEED=$(( i / 7 ))   # seeds 0-2
METHOD="${METHODS[$CFG]}"; MODSET="${MODULES[$CFG]}"; WMODE="${WHITEN[$CFG]}"
TAG="${TAGS[$CFG]}"; LR="${LRS[$CFG]}"
OUT_DIR="${OUT_ROOT:-results/imagenet100_noisy_deconfound_v1}/${TAG}_seed${SEED}"
mkdir -p "${OUT_DIR}"

COMMON=(--dataset imagenet1k --imagenet-root "${IMAGENET_ROOT}" --output-dir "${OUT_DIR}"
        --arch resnet18 --model-source torchvision --pretrained
        --epochs "${EPOCHS:-90}" --batch-size "${BATCH_SIZE:-256}" --workers 16
        --optimizer sgd --lr "${LR}" --momentum 0.9 --weight-decay 1e-4
        --warmup-epochs 5 --image-size 176 --val-resize 232
        --class-subset 100 --class-subset-seed 0 --seed "${SEED}"
        --label-noise "${LABEL_NOISE}")

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

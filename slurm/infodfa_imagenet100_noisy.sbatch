#!/usr/bin/env bash
# Experiment C: noisy-label ImageNet-100 head-to-head.
#
# Tests whether conditioned DFA exceeds BP under heavy label noise, mirroring
# Fashion-MNIST noisy (where we tied BP). ImageNet has higher feature-space
# anisotropy, which the linearized analysis says should help conditioning.
#
# Cell: ImageNet-100, 30% symmetric label noise (deterministic per --seed),
# 90 epochs, full data.
# Methods: bp, block_dfa at layer3+4, block_ndfa at layer3+4.
# Seeds: 3 per method.
#
# Total budget: 9 cells, each ~6h on 1 GPU.
#
#SBATCH --job-name=imgnet100_noisy
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --array=0-8%3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

METHODS=(bp block_dfa block_ndfa)
SEEDS=(0 1 2)
LABEL_NOISE="${LABEL_NOISE:-0.3}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_METHOD=${#METHODS[@]}
N_SEED=${#SEEDS[@]}

METHOD_IDX=$((TASK_ID / N_SEED))
SEED_IDX=$((TASK_ID % N_SEED))
METHOD="${METHODS[$METHOD_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

case "${METHOD}" in
  block_dfa|block_ndfa)
    FEEDBACK_MODULES="layer3,layer4"
    ;;
  *)
    FEEDBACK_MODULES="auto"
    ;;
esac

OUT_ROOT="results/imagenet100_noisy_v1"
OUT_DIR="${OUT_ROOT}/resnet18/${METHOD}/labelnoise_${LABEL_NOISE//./p}/seed_${SEED}"
mkdir -p "${OUT_DIR}"

python experiments/run_imagenet_credit_assignment.py \
  --dataset imagenet1k \
  --imagenet-root "${IMAGENET_ROOT:?set IMAGENET_ROOT to an ImageNet-1k ImageFolder}" \
  --output-dir "${OUT_DIR}" \
  --arch resnet18 \
  --model-source auto \
  --method "${METHOD}" \
  --feedback-modules "${FEEDBACK_MODULES}" \
  --feedback-scale 0.3 \
  --feedback-blend-gamma 1.0 \
  --feedback-spatial-mode broadcast \
  --dfa-norm bp \
  --natural-damping 0.1 \
  --grad-clip-norm 0.0 \
  --epochs 60 \
  --batch-size 128 \
  --workers 8 \
  --optimizer sgd \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --label-smoothing 0.0 \
  --warmup-epochs 5 \
  --image-size 224 \
  --val-resize 256 \
  --class-subset 100 \
  --class-subset-seed 0 \
  --label-noise "${LABEL_NOISE}" \
  --seed "${SEED}" \
  --amp \
  --channels-last

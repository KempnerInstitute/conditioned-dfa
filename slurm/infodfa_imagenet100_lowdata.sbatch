#!/usr/bin/env bash
# Experiment B: low-data ImageNet-100 head-to-head.
#
# Tests whether conditioned DFA exceeds BP at low data, mirroring the synthetic
# low-sample/noisy regime where nDFA beat BP by +14 pp.
#
# Cell: ImageNet-100 (--class-subset 100), 25 images/class (2500 total), 90 epochs.
# Methods: bp, block_dfa at layer3+4, block_ndfa at layer3+4.
# Seeds: 3 per method.
#
# Total budget: 9 cells. With 1 GPU each, ~2h/cell, ~9 GPU-hours total. A 4-GPU
# job will finish in ~2.5h.
#
#SBATCH --job-name=imgnet100_lowdata
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-8%4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

METHODS=(bp block_dfa block_ndfa)
SEEDS=(0 1 2)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_METHOD=${#METHODS[@]}
N_SEED=${#SEEDS[@]}

METHOD_IDX=$((TASK_ID / N_SEED))
SEED_IDX=$((TASK_ID % N_SEED))
METHOD="${METHODS[$METHOD_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

# Per-method module selection. BP uses no DFA modules.
case "${METHOD}" in
  block_dfa|block_ndfa)
    FEEDBACK_MODULES="layer3,layer4"
    ;;
  *)
    FEEDBACK_MODULES="auto"
    ;;
esac

OUT_ROOT="results/imagenet100_lowdata_v1"
OUT_DIR="${OUT_ROOT}/resnet18/${METHOD}/seed_${SEED}"
mkdir -p "${OUT_DIR}"

python experiments/run_imagenet_credit_assignment.py \
  --dataset imagenet1k \
  --imagenet-root "${IMAGENET_ROOT:-/n/holystore01/LABS/barak_lab/Everyone/datasets/imagenet}" \
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
  --warmup-epochs 5 \
  --image-size 224 \
  --val-resize 256 \
  --class-subset 100 \
  --class-subset-seed 0 \
  --subsample-train 2500 \
  --subsample-val 0 \
  --seed "${SEED}" \
  --amp \
  --channels-last

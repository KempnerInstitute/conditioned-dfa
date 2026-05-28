#!/usr/bin/env bash
#SBATCH --job-name=imgnet_weight_val
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-Geo}"
mkdir -p logs/slurm

python experiments/evaluate_imagenet_torchvision_weights.py \
  --imagenet-root "${IMAGENET_ROOT:-/n/holystore01/LABS/barak_lab/Everyone/datasets/imagenet}" \
  --output-dir "${OUT_DIR:-results/imagenet_torchvision_weight_validation_v1}" \
  --archs ${ARCHS:-alexnet resnet18 resnet50 convnext_tiny} \
  --batch-size "${BATCH_SIZE:-256}" \
  --workers "${WORKERS:-8}" \
  --subsample-val "${SUBSAMPLE_VAL:-0}" \
  --cuda

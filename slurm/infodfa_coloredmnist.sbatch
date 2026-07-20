#!/usr/bin/env bash
#SBATCH --job-name=infodfa_colormnist
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm results

python experiments/run_dfa_coloredmnist.py \
  --output-dir results/dfa_coloredmnist_v1 \
  --n-seeds 8 \
  --epochs 30 \
  --n-train 8000 \
  --n-val 2000 \
  --n-test 2000 \
  --train-color-correlation 0.85 \
  --hidden-dims 256 128 \
  --batch-size 128 \
  --lr 0.05 \
  --natural-damping 0.3 \
  --weight-decay 1e-3 \
  --download \
  --cuda

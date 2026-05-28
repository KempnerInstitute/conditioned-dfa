#!/usr/bin/env bash
#SBATCH --job-name=dfa_convscale_agg
#SBATCH --partition=serial_requeue
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-Geo}"
mkdir -p logs/slurm

python analysis/aggregate_dfa_convnet_baselines.py \
  --output-dir "${OUT_DIR:-results/dfa_convnet_scaled_aggregate_v1}" \
  --figure-name "${FIGURE_NAME:-infodfa_convnet_scaled}" \
  --inputs ${INPUTS:-"results/dfa_convnet_scaled_v1/*/mode_*/norm_*/lr_*/scale_*/*.csv"}

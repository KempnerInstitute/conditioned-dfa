#!/usr/bin/env bash
#SBATCH --job-name=infodfa_fact_agg
#SBATCH --partition=serial_requeue
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
shopt -s globstar nullglob

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

python analysis/aggregate_infodfa_factor_ablation.py \
  --synthetic-root "${SYNTHETIC_ROOT:-results/infodfa_factor_ablation_synthetic_v1}" \
  --vision-root "${VISION_ROOT:-results/infodfa_factor_ablation_vision_v1}" \
  --output-dir "${OUT_DIR:-results/infodfa_factor_ablation_aggregate_v1}"

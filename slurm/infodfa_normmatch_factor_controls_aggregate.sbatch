#!/usr/bin/env bash
#SBATCH --job-name=infodfa_normfac_agg
#SBATCH --partition=serial_requeue
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

python analysis/aggregate_infodfa_normmatch_factor_controls.py \
  --input-root "${INPUT_ROOT:-results/infodfa_normmatch_factor_controls_v1}" \
  --output-dir "${OUT_DIR:-results/infodfa_normmatch_factor_controls_aggregate_v1}"

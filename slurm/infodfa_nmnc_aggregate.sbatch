#!/usr/bin/env bash
#SBATCH --job-name=dfa_nmnc_agg
#SBATCH --partition=serial_requeue
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

python analysis/aggregate_dfa_nmnc_comparison.py \
  --output-dir "${OUT_DIR:-results/dfa_nmnc_aggregate_v1}" \
  --inputs ${INPUTS:-"results/dfa_nmnc_comparison_v1/*/b_*/rank_*/*.csv"}

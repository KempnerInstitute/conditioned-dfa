#!/usr/bin/env bash
# Aggregate Adam/diagonal-K approximation shards and make paper-facing figures.
#SBATCH --job-name=infodfa_adamk_agg
#SBATCH --partition=serial_requeue
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

OUT_ROOT="${OUT_ROOT:-results/infodfa_adam_diagk_approx_v1}"
OUT_DIR="${OUT_DIR:-results/infodfa_adam_diagk_aggregate_v1}"

python analysis/aggregate_infodfa_adam_diagk_approx.py \
  --inputs "${OUT_ROOT}"/shards/*/infodfa_adam_diagk_results.csv \
  --output-dir "${OUT_DIR}" \
  --preferred-damping "${PREFERRED_DAMPING:-0.3}" \
  --preferred-adaptive-lr "${PREFERRED_ADAPTIVE_LR:-0.003}"

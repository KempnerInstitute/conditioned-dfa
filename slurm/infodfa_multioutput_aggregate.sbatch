#!/usr/bin/env bash
#SBATCH --job-name=dfa_multiout_agg
#SBATCH --partition=serial_requeue
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

python analysis/aggregate_dfa_multioutput_synthetic.py \
  --output-dir "${OUT_DIR:-results/dfa_multioutput_synthetic_aggregate_v1}" \
  --inputs ${INPUTS:-"results/dfa_multioutput_synthetic_v1/shards/*/dfa_multioutput_results.csv"}

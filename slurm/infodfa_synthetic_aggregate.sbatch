#!/usr/bin/env bash
#SBATCH --job-name=dfa_synth_agg
#SBATCH --partition=serial_requeue
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

OUT_ROOT="${OUT_ROOT:-results/dfa_synthetic_full_array}"

python analysis/aggregate_dfa_synthetic_shards.py \
  --shard-root "${OUT_ROOT}/shards" \
  --output-dir "${OUT_ROOT}" \
  --early-epoch "${EARLY_EPOCH:-3}"

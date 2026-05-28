#!/usr/bin/env bash
#SBATCH --job-name=infodfa_noise_synth_agg
#SBATCH --partition=serial_requeue
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
shopt -s globstar nullglob

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-Geo}"
mkdir -p logs/slurm

OUT_ROOT="${OUT_ROOT:-results/infodfa_multioutput_noise_sweep_v1}"
AGG_DIR="${AGG_DIR:-results/infodfa_multioutput_noise_sweep_aggregate_v1}"

python analysis/aggregate_dfa_multioutput_synthetic.py \
  --inputs "${OUT_ROOT}"/**/dfa_multioutput_results.csv \
  --output-dir "${AGG_DIR}"

python analysis/make_infodfa_noise_sweep_figures.py \
  --multi-summary "${AGG_DIR}/dfa_multioutput_summary.csv" \
  --multi-final "${AGG_DIR}/dfa_multioutput_final_diagnostics.csv" \
  --output-dir "${FIG_DIR:-results/infodfa_noise_sweep_figures_v1}"

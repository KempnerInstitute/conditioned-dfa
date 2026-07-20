#!/usr/bin/env bash
#SBATCH --job-name=infodfa_c100_hard_agg
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

OUT_ROOT="${OUT_ROOT:-results/infodfa_hard_cifar100_confirm_v1}"
AGG_DIR="${AGG_DIR:-results/infodfa_hard_cifar100_confirm_aggregate_v1}"

python analysis/aggregate_dfa_convnet_baselines.py \
  --inputs "${OUT_ROOT}"/**/dfa_convnet_results.csv \
  --output-dir "${AGG_DIR}" \
  --figure-name infodfa_hard_cifar100_confirm

python analysis/make_infodfa_noise_sweep_figures.py \
  --conv-summary "${AGG_DIR}/dfa_convnet_summary.csv" \
  --conv-all "${AGG_DIR}/dfa_convnet_all.csv" \
  --output-dir "${FIG_DIR:-results/infodfa_noise_sweep_figures_v1}"

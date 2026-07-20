#!/usr/bin/env bash
#SBATCH --job-name=infodfa_bench
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/%x_%A.out
#SBATCH --error=logs/slurm/%x_%A.err

# Per-step wall-clock overhead of BP / DFA / nDFA / K-nDFA and amortized
# covariance refresh (k in {1,10,50}) on the paper's MLP configs, measured on
# both the allocated GPU and CPU. Backs the paper's runtime-overhead numbers
# (experiments/benchmark_overhead.py writes benchmark_overhead.csv per device).
set -euo pipefail
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2}"
cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

OUT_ROOT="${OUT_ROOT:-results/benchmark_overhead_v1}"

python experiments/benchmark_overhead.py \
  --device cuda \
  --output-dir "${OUT_ROOT}/cuda" \
  --warmup "${WARMUP:-20}" \
  --steps "${STEPS:-200}" \
  --refresh-intervals ${REFRESH_INTERVALS:-1 10 50} \
  --damping "${DAMPING:-0.3}"

python experiments/benchmark_overhead.py \
  --device cpu \
  --output-dir "${OUT_ROOT}/cpu" \
  --warmup "${WARMUP:-20}" \
  --steps "${STEPS:-200}" \
  --refresh-intervals ${REFRESH_INTERVALS:-1 10 50} \
  --damping "${DAMPING:-0.3}"

#!/usr/bin/env bash
#SBATCH --job-name=dfa_convcal_agg
#SBATCH --partition=serial_requeue
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

python analysis/aggregate_dfa_convnet_baselines.py \
  --output-dir "${OUT_DIR:-results/dfa_convnet_calibration_aggregate_v1}" \
  --inputs ${INPUTS:-"results/dfa_convnet_calibration_v1/cifar10/mode_*/norm_*/lr_*/scale_*/*.csv"}

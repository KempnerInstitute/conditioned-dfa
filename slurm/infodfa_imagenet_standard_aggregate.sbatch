#!/usr/bin/env bash
#SBATCH --job-name=imgnet_ca_agg
#SBATCH --partition=kempner_dev
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

DEFAULT_INPUTS="results/imagenet_credit_assignment_standard_v1/*/*/lr_*/scale_*/damping_*/seed_*/fb_*/*.csv results/imagenet_credit_assignment_standard_v1/*/*/modules_*/lr_*/scale_*/gamma_*/module_scales_*/damping_*/seed_*/fb_*/*.csv results/imagenet_credit_assignment_standard_v1/*/*/modules_*/spatial_*/lr_*/scale_*/gamma_*/module_scales_*/damping_*/seed_*/fb_*/*.csv"

python analysis/aggregate_imagenet_credit_assignment.py \
  --output-dir "${OUT_DIR:-results/imagenet_credit_assignment_standard_aggregate_v1}" \
  --figure-name "${FIGURE_NAME:-infodfa_imagenet_standard}" \
  --inputs ${INPUTS:-${DEFAULT_INPUTS}}

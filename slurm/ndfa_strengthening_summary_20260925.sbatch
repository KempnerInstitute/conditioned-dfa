#!/bin/bash
#SBATCH --account=kempner_dev
#SBATCH --partition=shared
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=ndfa_strength_summary
#SBATCH --output=logs/ndfa_strength_summary_%j.out
#SBATCH --error=logs/ndfa_strength_summary_%j.err
set -euo pipefail
: "${NDFA_SUMMARY_SCRIPT:?missing frozen analysis}"
: "${NDFA_SUMMARY_CONFIG:?missing study config}"
: "${NDFA_SUMMARY_OUTPUT:?missing output path}"
exec /n/sw/Mambaforge-23.11.0-0/bin/python "$NDFA_SUMMARY_SCRIPT" --config "$NDFA_SUMMARY_CONFIG" --output "$NDFA_SUMMARY_OUTPUT"

#!/bin/bash
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --job-name=ndfa_strength_next
#SBATCH --output=logs/ndfa_strength_next_%A_%a.out
#SBATCH --error=logs/ndfa_strength_next_%A_%a.err
set -euo pipefail
: "${NDFA_NEXT_SOURCE:?missing frozen source}"
: "${NDFA_NEXT_CONFIG:?missing config}"
: "${NDFA_NEXT_CONFIG_SHA256:?missing config hash}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$NDFA_NEXT_SOURCE"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /n/sw/Mambaforge-23.11.0-0/bin/python -u "$NDFA_NEXT_SOURCE/scripts/ndfa_strengthening_20260925/next_round.py" --config "$NDFA_NEXT_CONFIG" --config-sha256 "$NDFA_NEXT_CONFIG_SHA256"

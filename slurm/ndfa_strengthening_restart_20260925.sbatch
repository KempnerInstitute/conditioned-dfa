#!/bin/bash
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_requeue
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --requeue
#SBATCH --exclude=holygpu8a05402
#SBATCH --job-name=ndfa_strength_restart
#SBATCH --output=logs/ndfa_strength_restart_%A_%a.out
#SBATCH --error=logs/ndfa_strength_restart_%A_%a.err
set -euo pipefail
: "${NDFA_STRENGTH_SOURCE:?missing original frozen source}"
: "${NDFA_STRENGTH_CONFIG:?missing config}"
: "${NDFA_STRENGTH_CONFIG_SHA256:?missing config hash}"
: "${NDFA_RESTART_WRAPPER:?missing frozen wrapper}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$NDFA_STRENGTH_SOURCE"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /n/sw/Mambaforge-23.11.0-0/bin/python -u "$NDFA_RESTART_WRAPPER" --source "$NDFA_STRENGTH_SOURCE" --config "$NDFA_STRENGTH_CONFIG" --config-sha256 "$NDFA_STRENGTH_CONFIG_SHA256"

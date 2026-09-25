#!/bin/bash
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id38
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --exclude=holygpu8a05402
#SBATCH --job-name=ndfa_strength_timing
#SBATCH --output=logs/ndfa_strength_timing_%j.out
#SBATCH --error=logs/ndfa_strength_timing_%j.err
set -euo pipefail
: "${NDFA_STRENGTH_SOURCE:?missing frozen source}"
: "${NDFA_STRENGTH_OUTPUT:?missing output}"
: "${NDFA_STRENGTH_DATA:?missing data path}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$NDFA_STRENGTH_SOURCE"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /n/sw/Mambaforge-23.11.0-0/bin/python -u "$NDFA_STRENGTH_SOURCE/scripts/ndfa_strengthening_20260925/timing_pilot.py" --output "$NDFA_STRENGTH_OUTPUT" --data-dir "$NDFA_STRENGTH_DATA"

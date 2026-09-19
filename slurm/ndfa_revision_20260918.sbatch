#!/bin/bash
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id38
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:20:00
#SBATCH --no-requeue
#SBATCH --job-name=ndfa_revision
#SBATCH --output=logs/ndfa_revision_%j.out
#SBATCH --error=logs/ndfa_revision_%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /n/sw/Mambaforge-23.11.0-0/bin/python -u scripts/ndfa_revision_20260918/runner.py --plan configs/ndfa_revision_followups_20260918.json --phase "$1"

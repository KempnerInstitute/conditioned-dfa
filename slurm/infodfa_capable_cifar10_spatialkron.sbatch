#!/usr/bin/env bash
#SBATCH --job-name=infodfa_spk10
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --array=0
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Spatial-Kronecker conditioning (Eq. spatial_kron) on the capable CIFAR-10
# convnet. Adds a kernel-spatial covariance factor on top of channel-only nDFA;
# dropped into the SAME results dir as the capable CIFAR-10 run so it compares
# head-to-head with bp / ndfa_random / dfa_random under an identical recipe.
set -euo pipefail
# Optionally activate a conda env by exporting CONDA_ENV before submitting;
# otherwise the environment active at submission time is used.
if [ -n "${CONDA_ENV:-}" ]; then
    source "${CONDA_SH:-/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV}"
fi
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

METHOD=ndfa_spatial_kron
OUT_DIR="${OUT_ROOT:-results/infodfa_capable_cifar10_v1}/${METHOD}"
mkdir -p "${OUT_DIR}"

python experiments/run_dfa_convnet_baselines.py \
  --output-dir "${OUT_DIR}" --dataset cifar10 --download --methods "${METHOD}" \
  --n-seeds 5 --n-feedback-seeds 5 --n-train 50000 --n-test 10000 \
  --channels 64 128 256 --strides 1 2 2 --fc-hidden 512 \
  --epochs 40 --batch-size 128 --optimizer adam --lr 0.0001 \
  --weight-decay 1e-4 --feedback-scale 0.3 --feedback-mode channel \
  --feedback-normalization sqrt_flat --feedback-rank 0 --natural-damping 0.3 \
  --eval-size 1024 --eval-batch-size 512 --n-classes 10 --cuda --skip-analysis

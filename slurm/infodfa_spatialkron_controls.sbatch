#!/usr/bin/env bash
#SBATCH --job-name=infodfa_spkctl
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --array=0-5%3
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Robustness controls for the spatial-Kron both-directions result at alpha=1.0:
# (i) damping sweep (lambda in {0.1,1.0}; 0.3 is the headline run) to rule out
# mis-damping as the cause of D=acc(spatial-Kron)-acc(channel-nDFA)>0; and
# (ii) a k=8 spatial-scale check to show the effect tracks within-patch correlation,
# not one nuisance frequency. Each config runs channel-nDFA and spatial-Kron on the
# SAME corrupted data so D isolates the kernel-patch spatial factor.
set -euo pipefail
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2}"
cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

# config = (damping, k); methods are the two compared rules.
DAMPINGS=(0.1 0.1 1.0 1.0 0.3 0.3)
KS=(4 4 4 4 8 8)
METHODS=(ndfa_random ndfa_spatial_kron ndfa_random ndfa_spatial_kron ndfa_random ndfa_spatial_kron)
i="${SLURM_ARRAY_TASK_ID:-0}"
DAMP="${DAMPINGS[$i]}"; K="${KS[$i]}"; METHOD="${METHODS[$i]}"

OUT_DIR="results/infodfa_spatialkron_controls/d${DAMP}_k${K}/${METHOD}"
mkdir -p "${OUT_DIR}"

run_once() {
  python experiments/run_dfa_convnet_baselines.py \
    --output-dir "${OUT_DIR}" --dataset cifar10 --download --methods "${METHOD}" \
    --spatial-nuisance 1.0 --spatial-nuisance-scale "${K}" \
    --n-seeds 5 --n-feedback-seeds 5 --n-train 50000 --n-test 10000 \
    --channels 64 128 256 --strides 1 2 2 --fc-hidden 512 \
    --epochs 40 --batch-size 128 --optimizer adam --lr 0.0001 \
    --weight-decay 1e-4 --feedback-scale 0.3 --feedback-mode channel \
    --feedback-normalization sqrt_flat --feedback-rank 0 --natural-damping "${DAMP}" \
    --eval-size 1024 --eval-batch-size 512 --n-classes 10 --cuda --skip-analysis
}
for attempt in 1 2 3 4 5 6; do
  if run_once; then echo "succeeded on attempt ${attempt}"; break; fi
  echo "attempt ${attempt} failed (likely CUDA device-busy); sleeping before retry"
  sleep 90
done

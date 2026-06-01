#!/usr/bin/env bash
#SBATCH --job-name=infodfa_spknu100
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-15%6
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Generality check: replicate the spatial-Kronecker both-directions test on a SECOND
# dataset/architecture (capable CIFAR-100 convnet) to defang the "one dataset" critique.
# Self-contained matched sweep: alpha in {0,0.5,1.0,2.0} x {bp,dfa_random,ndfa_random,
# ndfa_spatial_kron}, identical capable recipe to the CIFAR-10 sweep. Predict the same
# sign flip: D=acc(spatial-Kron)-acc(channel-nDFA) rises from ~0/negative (clean) to
# positive as the class-independent low-frequency spatial nuisance grows.
set -euo pipefail
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate /n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2
cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

ALPHAS=(0.0 0.5 1.0 2.0)
METHODS=(bp dfa_random ndfa_random ndfa_spatial_kron)
i="${SLURM_ARRAY_TASK_ID:-0}"
ALPHA="${ALPHAS[$(( i / 4 ))]}"
METHOD="${METHODS[$(( i % 4 ))]}"

case "${METHOD}" in
  ndfa_random|ndfa_spatial_kron) LR=0.0001 ;;
  *)                             LR=0.0003 ;;  # bp, dfa_random
esac

OUT_DIR="results/infodfa_spatialkron_nuisance_cifar100/alpha${ALPHA}/${METHOD}"
mkdir -p "${OUT_DIR}"

run_once() {
  python experiments/run_dfa_convnet_baselines.py \
    --output-dir "${OUT_DIR}" --dataset cifar100 --download --methods "${METHOD}" \
    --spatial-nuisance "${ALPHA}" --spatial-nuisance-scale 4 \
    --n-seeds 5 --n-feedback-seeds 5 --n-train 50000 --n-test 10000 \
    --channels 64 128 256 --strides 1 2 2 --fc-hidden 512 \
    --epochs 40 --batch-size 128 --optimizer adam --lr "${LR}" \
    --weight-decay 1e-4 --feedback-scale 0.3 --feedback-mode channel \
    --feedback-normalization sqrt_flat --feedback-rank 0 --natural-damping 0.3 \
    --eval-size 1024 --eval-batch-size 512 --n-classes 100 --cuda --skip-analysis
}
# Retry the transient launch-time CUDA "device busy/unavailable" error on contended nodes.
for attempt in 1 2 3 4 5 6; do
  if run_once; then echo "succeeded on attempt ${attempt}"; break; fi
  echo "attempt ${attempt} failed (likely CUDA device-busy); sleeping before retry"
  sleep 90
done

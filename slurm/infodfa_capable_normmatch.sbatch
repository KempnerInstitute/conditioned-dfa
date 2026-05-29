#!/usr/bin/env bash
#SBATCH --job-name=infodfa_capnorm
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Capable-model whitening-vs-norm-match test (reviewer novelty crux).
# On a standard CIFAR convnet, compare covariance whitening (nDFA) against
# per-layer DFA gradient norm-matching to the BP norm. If nDFA > norm-match,
# whitening contributes beyond simple per-layer scale on a capable model.
set -euo pipefail
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate /n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2
cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

DATASETS=(cifar10 cifar100)
NCLASSES=(10 100)
i="${SLURM_ARRAY_TASK_ID:-0}"
DATASET="${DATASETS[$i]}"; NCLS="${NCLASSES[$i]}"
OUT="${OUT_ROOT:-results/infodfa_capable_normmatch_v1}/${DATASET}"
mkdir -p "${OUT}"

python experiments/run_dfa_convnet_baselines.py \
  --output-dir "${OUT}" --dataset "${DATASET}" --download \
  --methods bp dfa_random dfa_random_normmatch ndfa_random ndfa_random_kronecker local_loss \
  --n-seeds 3 --n-feedback-seeds 2 \
  --n-train 20000 --n-test 5000 \
  --channels 32 64 128 --strides 1 2 2 \
  --epochs 30 --batch-size 128 --lr 0.03 \
  --feedback-scale 1.0 --feedback-rank 0 --natural-damping 0.3 \
  --eval-size 512 --n-classes "${NCLS}" --feedback-mode channel --cuda --skip-analysis

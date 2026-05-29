#!/usr/bin/env bash
#SBATCH --job-name=infodfa_capnorm
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Capable-model whitening-vs-norm-match test (reviewer novelty crux), using the
# KNOWN-STABLE hard-CIFAR-100 convnet recipe (Adam, per-method LR, feedback-scale
# 0.3) so raw DFA does not diverge. dfa_random_normmatch uses the same LR as raw
# DFA (the `*)` case), giving a fair scale-only baseline against nDFA whitening.
set -euo pipefail
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate /n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2
cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

METHODS=(bp local_loss dfa_random dfa_random_normmatch ndfa_random ndfa_random_kronecker)
i="${SLURM_ARRAY_TASK_ID:-0}"
if (( i >= ${#METHODS[@]} )); then echo "Task ${i} >= ${#METHODS[@]}; exiting."; exit 0; fi
METHOD="${METHODS[$i]}"
OUT_DIR="${OUT_ROOT:-results/infodfa_capable_normmatch_v2}/${METHOD}"
mkdir -p "${OUT_DIR}"

case "${METHOD}" in
  bp)                                   LR_DEFAULT=0.0003 ;;
  local_loss)                           LR_DEFAULT=0.0001 ;;
  ndfa_random|ndfa_random_kronecker)    LR_DEFAULT=0.0001 ;;
  *)                                    LR_DEFAULT=0.0003 ;;  # dfa_random, dfa_random_normmatch
esac

python experiments/run_dfa_convnet_baselines.py \
  --output-dir "${OUT_DIR}" --dataset cifar100 --download --methods "${METHOD}" \
  --n-seeds 5 --n-feedback-seeds 5 --n-train 50000 --n-test 10000 \
  --channels 64 128 256 --strides 1 2 2 --fc-hidden 512 \
  --epochs 40 --batch-size 128 --optimizer adam --lr "${LR_DEFAULT}" \
  --weight-decay 1e-4 --feedback-scale 0.3 --feedback-mode channel \
  --feedback-normalization sqrt_flat --feedback-rank 0 --natural-damping 0.3 \
  --eval-size 1024 --eval-batch-size 512 --n-classes 100 --cuda --skip-analysis

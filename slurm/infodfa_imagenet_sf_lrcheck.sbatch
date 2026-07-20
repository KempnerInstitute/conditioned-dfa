#!/usr/bin/env bash
#SBATCH --job-name=infodfa_sf_lr
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# LR check for the strong-form (full-covariance) nDFA at the deep substitution
# depths where it collapsed in the main sweep (lr=0.1). Tests whether the deep
# collapse is an LR artifact of full-cov whitening (needs a smaller LR) or genuine.
set -euo pipefail
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2}"
cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm
export IMAGENET_ROOT="${IMAGENET_ROOT:-/n/holystore01/LABS/barak_lab/Everyone/datasets/imagenet}"

MODULES=("layer2,layer3,layer4" "layer1,layer2,layer3,layer4" "layer2,layer3,layer4" "layer1,layer2,layer3,layer4")
LRS=(0.03 0.03 0.01 0.01)
TAGS=(ndfaFull_l234_lr03 ndfaFull_all_lr03 ndfaFull_l234_lr01 ndfaFull_all_lr01)
i="${SLURM_ARRAY_TASK_ID:-0}"
OUT_DIR="${OUT_ROOT:-results/imagenet100_strongform_lrcheck_v1}/${TAGS[$i]}"; mkdir -p "${OUT_DIR}"

python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/run_imagenet_credit_assignment.py --method block_ndfa \
  --feedback-modules "${MODULES[$i]}" --feedback-seed 0 --feedback-spatial-mode broadcast \
  --dfa-norm unit --whiten-mode full --natural-damping 0.1 \
  --dataset imagenet1k --imagenet-root "${IMAGENET_ROOT}" --output-dir "${OUT_DIR}" \
  --arch resnet18 --model-source torchvision --pretrained \
  --epochs 90 --batch-size 256 --workers 16 --optimizer sgd --lr "${LRS[$i]}" \
  --momentum 0.9 --weight-decay 1e-4 --warmup-epochs 5 --image-size 176 --val-resize 232 \
  --class-subset 100 --class-subset-seed 0 --seed 0

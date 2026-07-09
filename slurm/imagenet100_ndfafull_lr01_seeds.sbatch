#!/usr/bin/env bash
#SBATCH --job-name=infodfa_in100_ndfafull_lr01
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-3%2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Missing seeds for the full-covariance ZCA ('ndfaFull') ImageNet-100 configs
# at lr 0.1: seeds 1 and 2 at substitution depths layer2+3+4 and all-block
# (4 runs). Seed 0 is the original strongform_v1 run
# (results/imagenet100_strongform_v1/ndfaFull_{l234,all}, lr 0.1); the existing
# multiseed extension reran these two depths at lr 0.01 only, so this array
# completes the lr-0.1 seed grid. Args replicate the seed-0 config.txt exactly,
# changing only the seed (and, per the multiseed convention, --feedback-seed).
set -euo pipefail
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate /n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2
cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

export IMAGENET_ROOT="${IMAGENET_ROOT:-/n/holystore01/LABS/barak_lab/Everyone/datasets/imagenet}"

MODULES=( "layer2,layer3,layer4" "layer1,layer2,layer3,layer4" )
TAGS=(    ndfaFull_l234_lr01 ndfaFull_all_lr01 )

i="${SLURM_ARRAY_TASK_ID:-0}"
CFG=$(( i % 2 ))
SEED=$(( i / 2 + 1 ))   # seeds 1 and 2
MODSET="${MODULES[$CFG]}"
TAG="${TAGS[$CFG]}"
LR="${LR:-0.1}"
OUT_DIR="${OUT_ROOT:-results/imagenet100_strongform_v1}/${TAG}_seed${SEED}"
mkdir -p "${OUT_DIR}"

COMMON=(--dataset imagenet1k --imagenet-root "${IMAGENET_ROOT}" --output-dir "${OUT_DIR}"
        --arch resnet18 --model-source torchvision --pretrained
        --epochs "${EPOCHS:-90}" --batch-size "${BATCH_SIZE:-256}" --workers 16
        --optimizer sgd --lr "${LR}" --momentum 0.9 --weight-decay 1e-4
        --warmup-epochs 5 --image-size 176 --val-resize 232
        --class-subset 100 --class-subset-seed 0 --seed "${SEED}")

run_once() {
  python -m torch.distributed.run --standalone --nproc_per_node=1 \
    experiments/run_imagenet_credit_assignment.py --method block_ndfa \
    --feedback-modules "$MODSET" --feedback-seed "${SEED}" --feedback-spatial-mode broadcast \
    --dfa-norm unit --whiten-mode full --natural-damping 0.1 "${COMMON[@]}"
}
for attempt in 1 2 3; do
  if run_once; then echo "succeeded on attempt ${attempt}"; break; fi
  echo "attempt ${attempt} failed (likely transient CUDA); sleeping before retry"
  sleep 90
done

#!/usr/bin/env bash
#SBATCH --job-name=infodfa_in100_sf
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-12%4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Strong-form ImageNet-100 substitution-depth sweep.
# Tests whether the FULL channel-covariance conditioner (--whiten-mode full)
# repairs the substitution-depth cliff that the diagonal form does not, with the
# BP-norm oracle dropped (--dfa-norm unit) and the backbone initialised from
# pretrained weights (matches the paper's fine-tuning design). Single seed.
set -euo pipefail
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2}"
cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

export IMAGENET_ROOT="${IMAGENET_ROOT:-/n/holystore01/LABS/barak_lab/Everyone/datasets/imagenet}"

#               0      1          2          3          4          5            6            7            8            9            10           11           12
METHODS=(       bp     block_dfa  block_dfa  block_dfa  block_dfa  block_ndfa   block_ndfa   block_ndfa   block_ndfa   block_ndfa   block_ndfa   block_ndfa   block_ndfa)
MODULES=(       none   layer4     "layer3,layer4" "layer2,layer3,layer4" "layer1,layer2,layer3,layer4" \
                       layer4     "layer3,layer4" "layer2,layer3,layer4" "layer1,layer2,layer3,layer4" \
                       layer4     "layer3,layer4" "layer2,layer3,layer4" "layer1,layer2,layer3,layer4")
WHITEN=(        diag   diag       diag       diag       diag       diag         diag         diag         diag         full         full         full         full)
TAGS=(          bp     dfa_layer4 dfa_l34    dfa_l234   dfa_all    ndfaDiag_layer4 ndfaDiag_l34 ndfaDiag_l234 ndfaDiag_all ndfaFull_layer4 ndfaFull_l34 ndfaFull_l234 ndfaFull_all)

i="${SLURM_ARRAY_TASK_ID:-0}"
METHOD="${METHODS[$i]}"; MODSET="${MODULES[$i]}"; WMODE="${WHITEN[$i]}"; TAG="${TAGS[$i]}"
OUT_DIR="${OUT_ROOT:-results/imagenet100_strongform_v1}/${TAG}"
mkdir -p "${OUT_DIR}"

COMMON=(--dataset imagenet1k --imagenet-root "${IMAGENET_ROOT}" --output-dir "${OUT_DIR}"
        --arch resnet18 --model-source torchvision --pretrained
        --epochs "${EPOCHS:-90}" --batch-size "${BATCH_SIZE:-256}" --workers 16
        --optimizer sgd --lr "${LR:-0.1}" --momentum 0.9 --weight-decay 1e-4
        --warmup-epochs 5 --image-size 176 --val-resize 232
        --class-subset 100 --class-subset-seed 0 --seed 0)

if [ "$METHOD" = "bp" ]; then
  python -m torch.distributed.run --standalone --nproc_per_node=1 \
    experiments/run_imagenet_credit_assignment.py --method bp "${COMMON[@]}"
else
  python -m torch.distributed.run --standalone --nproc_per_node=1 \
    experiments/run_imagenet_credit_assignment.py --method "$METHOD" \
    --feedback-modules "$MODSET" --feedback-seed 0 --feedback-spatial-mode broadcast \
    --dfa-norm unit --whiten-mode "$WMODE" --natural-damping 0.1 "${COMMON[@]}"
fi

#!/usr/bin/env bash
#SBATCH --job-name=dfa_imgnet
#SBATCH --partition=kempner_dev
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --array=0-4%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-Geo}"
mkdir -p logs/slurm

export OUT_ROOT="${OUT_ROOT:-results/dfa_convnet_imagenet100_pilot_v1}"
export DATASETS="${DATASETS:-imagenet100}"
export IMAGENET_ROOT="${IMAGENET_ROOT:-/n/holystore01/LABS/barak_lab/Everyone/datasets/imagenet}"
export DOWNLOAD="${DOWNLOAD:-0}"
export BALANCED_SUBSAMPLE="${BALANCED_SUBSAMPLE:-1}"
export IMAGE_SIZE="${IMAGE_SIZE:-64}"
export IMAGENET_CLASSES="${IMAGENET_CLASSES:-100}"
export CLASS_SUBSET_SEED="${CLASS_SUBSET_SEED:-0}"

export MODES="${MODES:-channel}"
export LRS="${LRS:-0.0001 0.0003}"
export BP_LRS="${BP_LRS:-0.0001 0.0003 0.001}"
export SCALES="${SCALES:-0.3}"
export METHODS="${METHODS:-dfa_random drtp_random ndfa_random ndfa_random_kronecker local_loss}"
export N_SEEDS="${N_SEEDS:-1}"
export N_FEEDBACK_SEEDS="${N_FEEDBACK_SEEDS:-1}"
export N_TRAIN="${N_TRAIN:-20000}"
export N_TEST="${N_TEST:-5000}"
export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
export CHANNELS="${CHANNELS:-96 192 384 384}"
export STRIDES="${STRIDES:-1 2 2 2}"
export KERNELS="${KERNELS:-3 3 3 3}"
export PADDINGS="${PADDINGS:-1 1 1 1}"
export FC_HIDDEN="${FC_HIDDEN:-2048}"
export OPTIMIZER="${OPTIMIZER:-adam}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
export EVAL_SIZE="${EVAL_SIZE:-1024}"

bash slurm/infodfa_convnet_scaled.sbatch

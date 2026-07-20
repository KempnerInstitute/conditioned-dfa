#!/usr/bin/env bash
#SBATCH --job-name=infodfa_actwhiten
#SBATCH --partition=serial_requeue
#SBATCH --time=06:00:00
#SBATCH --array=0-127%48
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Decorrelation baseline on the synthetic stress suite (does the gain need the
# full inverse?). dfa_actwhiten preconditions the DFA update by (C+lambda I)^{-1/2}
# (ZCA activation decorrelation, power 1/2) vs nDFA's full inverse (C+lambda I)^{-1}
# (natural-gradient power 1). Same 128-cell grid and recipe as the main sweep so the
# numbers are directly comparable to Table tab:infodfa_synthetic_noise_split.
set -euo pipefail

# Optionally activate a conda env by exporting CONDA_ENV before submitting;
# otherwise the environment active at submission time is used.
if [ -n "${CONDA_ENV:-}" ]; then
    source "${CONDA_SH:-/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV}"
fi

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs/slurm

CONDITIONS=(nuisance_dominant mixed_context low_sample_noisy task_aligned)
N_TRAINS=(512 1024 2048 4096)
LABEL_NOISES=(0.0 0.1 0.2 0.4)
INPUT_NOISES=(0.05 0.15)

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
N_NTRAINS=${#N_TRAINS[@]}; N_LABEL=${#LABEL_NOISES[@]}; N_INPUT=${#INPUT_NOISES[@]}
N_TOTAL=$(( ${#CONDITIONS[@]} * N_NTRAINS * N_LABEL * N_INPUT ))
if (( TASK_ID >= N_TOTAL )); then echo "Task ${TASK_ID} >= ${N_TOTAL}; exiting."; exit 0; fi

COND_INDEX=$((TASK_ID / (N_NTRAINS * N_LABEL * N_INPUT)))
REM=$((TASK_ID % (N_NTRAINS * N_LABEL * N_INPUT)))
NTRAIN_INDEX=$((REM / (N_LABEL * N_INPUT))); REM=$((REM % (N_LABEL * N_INPUT)))
LABEL_INDEX=$((REM / N_INPUT)); INPUT_INDEX=$((REM % N_INPUT))

CONDITION="${CONDITIONS[$COND_INDEX]}"
N_TRAIN_CELL="${N_TRAINS[$NTRAIN_INDEX]}"
LABEL_NOISE="${LABEL_NOISES[$LABEL_INDEX]}"
INPUT_NOISE="${INPUT_NOISES[$INPUT_INDEX]}"
label_tag="${LABEL_NOISE//./p}"; input_tag="${INPUT_NOISE//./p}"
OUT_BASE="${OUT_ROOT:-results/infodfa_actwhiten_synthetic_v1}"
OUT_DIR="${OUT_BASE}/${CONDITION}/ntrain_${N_TRAIN_CELL}/label_${label_tag}/input_${input_tag}"
mkdir -p "${OUT_DIR}"

python experiments/run_dfa_multioutput_synthetic.py \
  --output-dir "${OUT_DIR}" \
  --conditions "${CONDITION}" \
  --methods dfa_random dfa_actwhiten ndfa_random ndfa_random_kronecker \
  --n-seeds 5 --n-feedback-seeds 5 --feedback-ranks 0 \
  --n-train "${N_TRAIN_CELL}" --n-test 4096 \
  --input-dim 64 --n-classes 8 --nuisance-dim 24 \
  --input-noise "${INPUT_NOISE}" --label-noise "${LABEL_NOISE}" \
  --hidden-dims 256 128 --epochs 14 --batch-size 128 --lr 0.08 \
  --eval-size 512 --pca-size 1024 --natural-damping 0.3 --skip-analysis

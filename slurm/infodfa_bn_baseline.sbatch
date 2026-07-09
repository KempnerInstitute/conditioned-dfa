#!/usr/bin/env bash
#SBATCH --job-name=infodfa_bn_base
#SBATCH --partition=serial_requeue
#SBATCH --time=10:00:00
#SBATCH --array=0-139%32
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# DFA+BatchNorm and BP+BatchNorm baselines (reviewer ask: does BatchNorm's
# implicit activation normalization already close the DFA gap that nDFA
# conditioning closes?). --batchnorm inserts BatchNorm1d after each hidden
# linear layer (pre-activation) in the manual MLP; all other protocol args
# match the paper sweeps.
#   Tasks 0-127: the 128-cell synthetic stress suite
#     (run_dfa_multioutput_synthetic.py grid of infodfa_multioutput_noise_sweep.sbatch).
#   Tasks 128-139: noisy Fashion-MNIST, n_train {1000,3000,10000} x
#     label-noise {0,0.1,0.2,0.4} (run_dfa_nmnc_comparison.py grid of
#     infodfa_vision_noise_sweep.sbatch).
# Methods are bp and dfa_random at full-rank feedback (rank 0); the paper's
# honest rank selection found full rank essentially always best, and the
# comparison partner columns (BP/DFA/nDFA without BN) already exist in the
# main sweeps. Override METHODS/FEEDBACK_RANKS to widen.
set -euo pipefail

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
OUT_ROOT="${OUT_ROOT:-results/infodfa_bn_baseline_v1}"

if (( TASK_ID < 128 )); then
  # --- synthetic suite cell ---
  CONDITIONS=(${CONDITIONS:-nuisance_dominant mixed_context low_sample_noisy task_aligned})
  N_TRAINS=(${N_TRAINS:-512 1024 2048 4096})
  LABEL_NOISES=(${LABEL_NOISES:-0.0 0.1 0.2 0.4})
  INPUT_NOISES=(${INPUT_NOISES:-0.05 0.15})

  N_NTRAINS=${#N_TRAINS[@]}
  N_LABEL=${#LABEL_NOISES[@]}
  N_INPUT=${#INPUT_NOISES[@]}

  COND_INDEX=$((TASK_ID / (N_NTRAINS * N_LABEL * N_INPUT)))
  REM=$((TASK_ID % (N_NTRAINS * N_LABEL * N_INPUT)))
  NTRAIN_INDEX=$((REM / (N_LABEL * N_INPUT)))
  REM=$((REM % (N_LABEL * N_INPUT)))
  LABEL_INDEX=$((REM / N_INPUT))
  INPUT_INDEX=$((REM % N_INPUT))

  CONDITION="${CONDITIONS[$COND_INDEX]}"
  N_TRAIN_CELL="${N_TRAINS[$NTRAIN_INDEX]}"
  LABEL_NOISE="${LABEL_NOISES[$LABEL_INDEX]}"
  INPUT_NOISE="${INPUT_NOISES[$INPUT_INDEX]}"
  label_tag="${LABEL_NOISE//./p}"
  input_tag="${INPUT_NOISE//./p}"
  OUT_DIR="${OUT_ROOT}/synthetic/${CONDITION}/ntrain_${N_TRAIN_CELL}/label_${label_tag}/input_${input_tag}"
  mkdir -p "${OUT_DIR}"

  python experiments/run_dfa_multioutput_synthetic.py \
    --output-dir "${OUT_DIR}" \
    --conditions "${CONDITION}" \
    --methods ${METHODS:-bp dfa_random} \
    --batchnorm \
    --n-seeds "${N_SEEDS:-5}" \
    --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
    --feedback-ranks ${FEEDBACK_RANKS:-0} \
    --n-train "${N_TRAIN_CELL}" \
    --n-test "${N_TEST:-4096}" \
    --input-dim "${INPUT_DIM:-64}" \
    --n-classes "${N_CLASSES:-8}" \
    --nuisance-dim "${NUISANCE_DIM:-24}" \
    --input-noise "${INPUT_NOISE}" \
    --label-noise "${LABEL_NOISE}" \
    --hidden-dims ${HIDDEN_DIMS:-256 128} \
    --epochs "${EPOCHS:-14}" \
    --batch-size "${BATCH_SIZE:-128}" \
    --lr "${LR:-0.08}" \
    --feedback-scale "${FEEDBACK_SCALE:-1.0}" \
    --natural-damping "${NATURAL_DAMPING:-0.3}" \
    --eval-size "${EVAL_SIZE:-512}" \
    --pca-size "${PCA_SIZE:-1024}" \
    --covariance-diagnostics \
    --checkpoint-every "${CHECKPOINT_EVERY:-8}" \
    --skip-analysis
else
  # --- noisy Fashion-MNIST cell ---
  VIS_N_TRAINS=(${VIS_N_TRAINS:-1000 3000 10000})
  VIS_LABEL_NOISES=(${VIS_LABEL_NOISES:-0.0 0.1 0.2 0.4})
  VIS_ID=$((TASK_ID - 128))
  N_VIS_LABEL=${#VIS_LABEL_NOISES[@]}
  N_VIS_TOTAL=$(( ${#VIS_N_TRAINS[@]} * N_VIS_LABEL ))
  if (( VIS_ID >= N_VIS_TOTAL )); then
    echo "Task ${TASK_ID} beyond vision grid; exiting."
    exit 0
  fi
  N_TRAIN_CELL="${VIS_N_TRAINS[$((VIS_ID / N_VIS_LABEL))]}"
  LABEL_NOISE="${VIS_LABEL_NOISES[$((VIS_ID % N_VIS_LABEL))]}"
  label_tag="${LABEL_NOISE//./p}"
  OUT_DIR="${OUT_ROOT}/fashion_mnist/ntrain_${N_TRAIN_CELL}/label_${label_tag}"
  mkdir -p "${OUT_DIR}"

  python experiments/run_dfa_nmnc_comparison.py \
    --output-dir "${OUT_DIR}" \
    --dataset fashion_mnist \
    --download \
    --methods ${METHODS:-bp dfa_random} \
    --batchnorm \
    --n-seeds "${N_SEEDS:-5}" \
    --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
    --n-train "${N_TRAIN_CELL}" \
    --n-test "${N_TEST:-2000}" \
    --label-noise "${LABEL_NOISE}" \
    --hidden-dims ${HIDDEN_DIMS:-512 256} \
    --epochs "${EPOCHS:-12}" \
    --batch-size "${BATCH_SIZE:-128}" \
    --lr "${LR:-0.08}" \
    --feedback-scale "${FEEDBACK_SCALE:-1.0}" \
    --feedback-rank "${FEEDBACK_RANK:-0}" \
    --natural-damping "${NATURAL_DAMPING:-0.3}" \
    --eval-size "${EVAL_SIZE:-512}" \
    --pca-size "${PCA_SIZE:-1024}" \
    --local-pca-neighbors "${LOCAL_PCA_NEIGHBORS:-24}" \
    --local-pca-rank "${LOCAL_PCA_RANK:-8}" \
    --covariance-diagnostics
fi

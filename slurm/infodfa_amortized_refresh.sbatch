#!/usr/bin/env bash
#SBATCH --job-name=infodfa_amort_refresh
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --time=10:00:00
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Amortized covariance refresh: does recomputing the damped inverse only every
# k batches (--cov-refresh-interval, cached inverse reused in between) preserve
# the nDFA/K-nDFA accuracy gains while cutting the conditioning overhead
# (timed separately by benchmark_overhead.sbatch)? k=1 is the exact-refresh
# anchor identical to the paper protocol.
#
# Cells: the nuisance-dominant focused synthetic cell of the factor ablation
# (nuisance_dominant, n_train 512, label-noise 0.2, input-noise 0.15) and the
# noisy Fashion-MNIST headline cell (n_train 10000, label-noise 0.4);
# k in {1,10,50}; 5 data seeds x 3 feedback seeds; otherwise the paper's
# sweep protocol.
set -euo pipefail

source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate /n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2

cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

REFRESH_KS=(${REFRESH_KS:-1 10 50})
CELLS=(synthetic_nuisance fashion_mnist_noisy)

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
N_KS=${#REFRESH_KS[@]}
CELL="${CELLS[$((TASK_ID / N_KS))]}"
REFRESH_K="${REFRESH_KS[$((TASK_ID % N_KS))]}"

OUT_ROOT="${OUT_ROOT:-results/infodfa_amortized_refresh_v1}"
OUT_DIR="${OUT_ROOT}/${CELL}/k_${REFRESH_K}"
mkdir -p "${OUT_DIR}"

if [ "${CELL}" = "synthetic_nuisance" ]; then
  python experiments/run_dfa_multioutput_synthetic.py \
    --output-dir "${OUT_DIR}" \
    --conditions nuisance_dominant \
    --methods ${METHODS:-bp dfa_random ndfa_random ndfa_random_kronecker} \
    --cov-refresh-interval "${REFRESH_K}" \
    --n-seeds "${N_SEEDS:-5}" \
    --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
    --feedback-ranks ${FEEDBACK_RANKS:-0} \
    --n-train "${N_TRAIN:-512}" \
    --n-test "${N_TEST:-4096}" \
    --input-dim "${INPUT_DIM:-64}" \
    --n-classes "${N_CLASSES:-8}" \
    --nuisance-dim "${NUISANCE_DIM:-24}" \
    --input-noise "${INPUT_NOISE:-0.15}" \
    --label-noise "${LABEL_NOISE:-0.2}" \
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
  python experiments/run_dfa_nmnc_comparison.py \
    --output-dir "${OUT_DIR}" \
    --dataset fashion_mnist \
    --download \
    --methods ${METHODS:-bp dfa_random ndfa_random ndfa_random_kronecker} \
    --cov-refresh-interval "${REFRESH_K}" \
    --n-seeds "${N_SEEDS:-5}" \
    --n-feedback-seeds "${N_FEEDBACK_SEEDS:-3}" \
    --n-train "${N_TRAIN:-10000}" \
    --n-test "${N_TEST:-2000}" \
    --label-noise "${LABEL_NOISE:-0.4}" \
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

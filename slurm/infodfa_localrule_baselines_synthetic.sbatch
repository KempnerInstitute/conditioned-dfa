#!/usr/bin/env bash
#SBATCH --job-name=infodfa_localrule
#SBATCH --partition=serial_requeue
#SBATCH --time=06:00:00
#SBATCH --array=0-127%48
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Local-rule baselines on the synthetic suite: add sign-symmetry (fa_sign) as a
# comparator alongside the existing FA/DFA/DRTP and conditioned-DFA family.
# (Kolen-Pollack and PEPITA require persistent-feedback / two-pass machinery and
# are tracked as a separate follow-up.) METHODS overridable.
set -euo pipefail
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate /n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2
cd "${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/Info-DFA}"
mkdir -p logs/slurm

CONDITIONS=(nuisance_dominant mixed_context low_sample_noisy task_aligned)
N_TRAINS=(512 1024 2048 4096); LABEL_NOISES=(0.0 0.1 0.2 0.4); INPUT_NOISES=(0.05 0.15)
i="${SLURM_ARRAY_TASK_ID:-0}"
N_NT=${#N_TRAINS[@]}; N_LB=${#LABEL_NOISES[@]}; N_IN=${#INPUT_NOISES[@]}
N_TOTAL=$(( ${#CONDITIONS[@]} * N_NT * N_LB * N_IN ))
if (( i >= N_TOTAL )); then echo "Task ${i} >= ${N_TOTAL}; exiting."; exit 0; fi
ci=$((i / (N_NT*N_LB*N_IN))); r=$((i % (N_NT*N_LB*N_IN)))
ni=$((r / (N_LB*N_IN))); r=$((r % (N_LB*N_IN))); li=$((r / N_IN)); ii=$((r % N_IN))
COND="${CONDITIONS[$ci]}"; NT="${N_TRAINS[$ni]}"; LB="${LABEL_NOISES[$li]}"; IN="${INPUT_NOISES[$ii]}"
lt="${LB//./p}"; it="${IN//./p}"
OUT="${OUT_ROOT:-results/infodfa_localrule_baselines_v1}/${COND}/ntrain_${NT}/label_${lt}/input_${it}"; mkdir -p "${OUT}"

python experiments/run_dfa_multioutput_synthetic.py \
  --output-dir "${OUT}" --conditions "${COND}" \
  --methods ${METHODS:-bp dfa_random fa_random fa_sign ndfa_random ndfa_random_kronecker} \
  --n-seeds 5 --n-feedback-seeds 1 --feedback-ranks 0 \
  --n-train "${NT}" --n-test 4096 --input-dim 64 --n-classes 8 --nuisance-dim 24 \
  --input-noise "${IN}" --label-noise "${LB}" \
  --hidden-dims 256 128 --epochs 14 --batch-size 128 --lr 0.08 \
  --natural-damping 0.3 --eval-size 512 --pca-size 1024 --skip-analysis

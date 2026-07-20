# Reproducing the results

Every numerical claim in the paper is tied to a script in `experiments/` (or a
`slurm/*.sbatch` array), an aggregator in `analysis/`, and either a committed
table/figure snapshot or an aggregate CSV under `results/`. Aggregate CSVs land
under `results/` (gitignored; sizes can be large), so a fresh checkout may need
those roots restored from the artifact bundle or regenerated before table/stat
scripts can be rerun. This file lists the command, grid, aggregator, and artifact
root for each result. Final figure assembly and the LaTeX build live in the
separately distributed paper source (included with the arXiv submission).

## Current claim boundary

- The primary synthetic comparison fixes full-rank feedback. Test-selected rank
  summaries are retained only as historical sensitivity analyses.
- The method family contains activity nDFA, error nDFA, and two-sided K-nDFA.
  Historical error-side/two-sided results used deltas already divided by the
  mean-loss batch factor and remain excluded. The corrected DFA-stall evidence
  uses per-example errors, independently validation-selected activity/error
  damping, norm matching, and frozen confirmations on clean MNIST and
  Fashion-MNIST, each with five model/data-order seeds crossed with three
  feedback seeds. The Fashion-MNIST run was preregistered and includes a
  corrected nonlocal BP-error covariance-source control.
- A fresh-seed architectural confirmation uses a 256--128 ReLU MLP with
  multiclass softmax cross-entropy. It independently selects both dampings on
  validation data and confirms the error-only and incremental K-nDFA signs on
  eight model/data-order seeds crossed with three feedback seeds. A negative
  exploratory ReLU Fashion-MNIST development pilot limits the claim to MNIST.
- The ImageNet-100 experiment applies diagonal/full inverse-square-root
  conditioning to pooled block outputs. It is a separate credit-assignment
  diagnostic, not Eq. (3)'s activity-side nDFA update.
- Cell-level and crossed-seed intervals are descriptive. The five global data
  seeds are the replication unit for the conservative synthetic sensitivity
  analysis.

## Environment
- Python 3.10. Install dependencies with `pip install -r requirements.txt` (torch 2.9,
  torchvision 0.24, timm 1.0, numpy, pandas, scipy, matplotlib), then activate that
  environment before any run; any environment satisfying `requirements.txt`
  reproduces the results.
- SLURM scripts set `REPO_ROOT` to this repo and `IMAGENET_ROOT` to an
  ImageNet ImageFolder; override either by env var. The `#SBATCH` headers
  (partition, account) and the conda activation line are site-specific: on
  other systems edit the headers, export `CONDA_ENV`, and set `REPO_ROOT`
  before submitting.
- ImageNet-1k must be obtained separately (standard ImageFolder train/val
  layout) and pointed to via `IMAGENET_ROOT`; the ImageNet-100 subset is
  selected in-code by `--class-subset 100 --class-subset-seed 0`. MNIST,
  Fashion-MNIST, and CIFAR download automatically.

## Claim-to-artifact map

| Claim/table/figure | Script or table source | Required artifact root |
| --- | --- | --- |
| Table 1 fixed-full synthetic rows and seed sensitivity | `analysis/reanalyze_synthetic_honest_selection.py`, embedded revised TeX table | `results/infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_all.csv` |
| Table 1 exploratory noisy-vision rows | `analysis/write_infodfa_paper_tables.py` | `results/infodfa_vision_noise_sweep_aggregate_v2/dfa_nmnc_best_by_method.csv` |
| Table 1 and Appendix activity/error/K-nDFA confirmation | `experiments/run_dfa_stall_comparison.py`, `analysis/analyze_dfa_stall_threefactor.py` | `results/dfa_stall_threefactor_dev_v1`, `results/dfa_stall_threefactor_confirmation_v1` |
| Table 1 and Appendix clean Fashion-MNIST replication/source swap | `slurm/infodfa_dfa_stall_fashion_threefactor.sbatch`, `analysis/analyze_dfa_stall_fashion_threefactor.py` | `results/dfa_stall_fashion_threefactor_{dev,confirmation,analysis}_v1` |
| Table 1, Fig. 3, and Appendix ReLU/softmax confirmation | `slurm/infodfa_dfa_relu_mnist_threefactor.sbatch`, `analysis/analyze_dfa_relu_vision_threefactor.py`, `analysis/make_error_kndfa_replication_figure.py` | `results/dfa_relu_mnist_{dev,threefactor_confirmation,threefactor_analysis}_v1` |
| Appendix BP-source scale diagnosis and post-hoc retuning | `analysis/diagnose_dfa_stall_bpsource_scale.py`, `slurm/infodfa_dfa_stall_fashion_bpsource_retune.sbatch`, `analysis/analyze_dfa_stall_bpsource_retune.py` | `results/dfa_stall_fashion_bpsource_scale_audit_v1`, `results/dfa_stall_fashion_bpsource_scale_audit_d3_v1`, `results/dfa_stall_fashion_bpsource_retune_{dev,confirmation,analysis}_v1` |
| Main figures | figure-assembly script shipped with the paper source (arXiv submission) | aggregate roots listed in the sections below |
| Activity norm-match, Adam/diagonal, decorrelation, BatchNorm, and BP-precondition controls | corresponding `analysis/aggregate_*.py` scripts below | control-specific roots listed below |
| ImageNet-100 boundary table | `analysis/aggregate_imagenet_strongform.py` plus the hand-curated table snapshot | `results/imagenet100_strongform_v1/strongform_multiseed_summary.csv` |

`results/` is not versioned: before rerunning table/stat scripts, regenerate
the aggregate CSVs (in particular the synthetic and noisy-vision
`*_best_by_method.csv` files) with the SLURM commands below. Regenerated table
snapshots are written under
`results/infodfa_paper_tables_20260527/tex_tables/` by default.

## Paper tables

The table scripts resolve artifacts from a primary results root and an optional
secondary root:

```bash
export INFODFA_RESULTS=/path/to/Info-DFA/results
export INFODFA_LEGACY_RESULTS=/path/to/legacy/results  # optional secondary root
python analysis/reanalyze_synthetic_honest_selection.py
python analysis/write_infodfa_paper_tables.py
```

`analysis/write_infodfa_paper_tables.py` records the resolved table inputs in
`results/infodfa_paper_tables_20260527/infodfa_final_result_summary.md` and
writes regenerable TeX snapshots under
`results/infodfa_paper_tables_20260527/tex_tables/` unless
`INFODFA_TABLE_TEX_DIR` is set. Final figure assembly and the LaTeX build live
in the paper source distributed with the arXiv submission (shared body
`paper_body.tex` with two selectable roots, `conditioned_dfa_arxiv.tex` and
`conditioned_dfa_iclr.tex`); the paper build uses committed figure files.

## Synthetic stress suite (Tables 1, 10; Fig 2)
- Run: `sbatch slurm/infodfa_multioutput_noise_sweep.sbatch` (128 cells = 4 regimes
  x 4 n_train {512,1024,2048,4096} x 4 label-noise {0,0.1,0.2,0.4} x 2 input-noise
  {0.05,0.15}; methods bp/dfa_random/fa_random/ndfa_random/ndfa_random_kronecker/
  drtp_random/vnc/nmnc; 5 data seeds; 3 feedback seeds for local-feedback methods;
  feedback ranks {0,1,2,4,8}; hidden 256-128; 14 training epochs plus epoch-0
  evaluation; lr 0.08; damping `lambda=0.3`).
- Honest model selection: `analysis/reanalyze_synthetic_honest_selection.py` reports
  per-regime accuracy under test-selected vs fixed-full-rank vs leave-one-seed-out
  (LOSO) rank selection, plus matched cell-by-seed paired robustness tests. These
  tests retain held-out seed identity but are not five independent seed-level
  inference. The three selection schemes agree to <0.1pp (full-rank feedback is
  essentially always best).
- Conservative sensitivity analysis: `analysis/compute_infodfa_seedlevel_stats.py`
  averages the fixed-full-rank nDFA--DFA difference within each of the five
  global data seeds. Its bootstrap intervals are descriptive because the
  experimental grid is fixed rather than sampled from a population of tasks.

## Corrected activity/error/K-nDFA confirmation

- Run `sbatch slurm/infodfa_dfa_stall_threefactor.sbatch`. Tasks 0--5 select
  activity damping for activity nDFA; tasks 6--11 independently select error
  damping for error nDFA; task 12 checks raw DFA and the combined rule on the
  development seeds; tasks 13--15 run the frozen four-method confirmation, one
  task per feedback seed.
- The fixed protocol uses a width-300, three-hidden-layer tanh MNIST MLP, 1,000
  steps, layerwise norm matching, a 5,000-example training-validation split,
  development seeds 42--44, and confirmation seeds 50--54 crossed with feedback
  seeds 0--2. The test set is evaluated only after the last update.
- After the array completes, run
  `python analysis/analyze_dfa_stall_threefactor.py`. It verifies
  `lambda_A=0.3` and `lambda_E=10`, averages feedback seeds within model seed,
  writes seed-level contrasts under `results/dfa_stall_threefactor_analysis_v1`,
  and regenerates `figures/iclr_fig_threefactor_conditioning.{pdf,png,svg}`.

### Preregistered clean Fashion-MNIST replication and source-swap diagnosis

- The protocol and pass/fail criteria are recorded in `PREDICTIONS.md` at
  commit `ef795e1`, before the development and confirmation runs. Run tasks
  0--15 of
  `slurm/infodfa_dfa_stall_fashion_threefactor.sbatch` for the independent
  activity/error development sweeps. They freeze `lambda_A=0.03` and
  `lambda_E=30` in commit `f2eaf3a`; run tasks 16--19 only from that frozen
  version.
- The clean Fashion-MNIST protocol uses the same width-300 tanh MLP, 1,000
  steps, norm matching, and final-only test evaluation. Development uses split
  seed 24680, model/data-order seeds 60--62, one feedback seed, and damping grid
  `{0.03,0.1,0.3,1,3,10,30,100}`. Confirmation uses seeds 70--74 crossed with
  feedback seeds 0--2. `kndfa_bp` changes only the K-nDFA error covariance from
  local DFA hidden errors to exact BP hidden errors; it is a nonlocal
  source-swap comparator, not BP training.
- Run `python analysis/analyze_dfa_stall_fashion_threefactor.py`. It writes the
  endpoint table, paired contrasts, preregistration scorecard, and
  `iclr_fig_fashion_threefactor_conditioning.{pdf,png,svg}` under
  `results/dfa_stall_fashion_threefactor_analysis_v1` and the paper figure root.
- The registered BP-source comparator is not a live source-equivalence test at
  the frozen local damping. Run
  `python analysis/diagnose_dfa_stall_bpsource_scale.py` to reproduce the
  layerwise spectral-scale and update-cosine audit (defaults: `--error-damping
  30`, output `results/dfa_stall_fashion_bpsource_scale_audit_v1`); rerun with
  `--error-damping 3 --output-dir
  results/dfa_stall_fashion_bpsource_scale_audit_d3_v1` for the retuned-damping
  variant.
- For the explicitly post-hoc scale-matched addendum, run development tasks
  0--9 of `slurm/infodfa_dfa_stall_fashion_bpsource_retune.sbatch`. Validation
  selects BP-source damping 3 (tied accuracy with 10, lower loss). Then submit
  tasks 10--12 with `BP_ERROR_DAMPING=3`; these use fresh model/data-order seeds
  80--84 crossed with feedback seeds 0--2. Aggregate with
  `python analysis/analyze_dfa_stall_bpsource_retune.py`.

### Fresh-seed ReLU/softmax architectural replication

- The P7 protocol was frozen in `PREDICTIONS.md` before confirmation test
  evaluation. It uses clean MNIST, a 256--128 ReLU MLP, softmax cross-entropy,
  1,000 updates, batch size 128, learning rate 0.03, and layerwise hidden-weight
  gradient norm matching.
- Run `sbatch slurm/infodfa_dfa_relu_mnist_threefactor.sbatch`. Tasks 0--14
  perform the independent validation grids, task 15 records the selected joint
  development check, and tasks 16--18 run the final-only-test confirmation.
- On split seed 86420, run activity-nDFA development seeds 0--2 over
  `{0.03,0.1,0.3,1,3,10,30}` and error-nDFA over
  `{0.003,0.01,0.03,0.1,0.3,1,3,10}` with test evaluation disabled. The
  validation rule selects `lambda_A=3` and `lambda_E=0.1`; K-nDFA combines them
  without joint tuning.
- Run the four methods with those frozen values on model/data-order seeds
  100--107 separately for feedback seeds 0, 1, and 2, enabling final-only test
  evaluation. Then run:

  ```bash
  python analysis/analyze_dfa_relu_vision_threefactor.py
  python analysis/make_error_kndfa_replication_figure.py
  ```

  The first command verifies the frozen selection, averages feedback seeds
  within model seed, scores P7, and regenerates the appendix figure. The second
  combines the tanh MNIST, tanh Fashion-MNIST, and ReLU MNIST paired contrasts
  into the main error/K-nDFA replication figure.

## Historical activity/error factor ablation (two-sided rows are not evidence)
- Purpose: isolate which Kronecker side drives the first-paper gains. The comparison
  is BP, DFA, activity-only nDFA (`ndfa_random`), error-only nDFA
  (`ndfa_random_error`), and full K-nDFA (`ndfa_random_kronecker`).
- Synthetic focused cells: `sbatch slurm/infodfa_factor_ablation_synthetic.sbatch`
  (nuisance-dominant, mixed-context, low-sample noisy, and clean task-aligned cells;
  5 data seeds x 3 feedback seeds; feedback rank 0; hidden 256-128; 100 epochs by
  default for saturation curves).
- Vision focused cells: `sbatch slurm/infodfa_factor_ablation_vision.sbatch`
  (Fashion-MNIST and CIFAR-10 with n_train {1000,3000}, label-noise 0.4; same
  method set and 100-epoch default).
- Aggregate after both arrays finish: `sbatch slurm/infodfa_factor_ablation_aggregate.sbatch`,
  or directly:
  `python analysis/aggregate_infodfa_factor_ablation.py --synthetic-root
  results/infodfa_factor_ablation_synthetic_v1 --vision-root
  results/infodfa_factor_ablation_vision_v1 --output-dir
  results/infodfa_factor_ablation_aggregate_v1`.
- Main outputs: `infodfa_factor_ablation_curves.csv`,
  `infodfa_factor_ablation_endpoints.csv`,
  `infodfa_factor_ablation_effect_curves.csv`,
  `infodfa_factor_ablation_effects.csv`, `infodfa_factor_ablation_summary.md`,
  and publication-ready curve/endpoint/factor-decomposition figures in png/pdf/svg.

## Norm-matched controls (paper uses activity-side rows only)
- Purpose: test whether activity/error/K-nDFA effects are merely scalar
  learning-rate or per-layer gradient-norm changes. The `+norm_match` variants
  rescale each local hidden gradient to a matched BP layer norm after applying
  the indicated conditioner.
- Run: `sbatch slurm/infodfa_normmatch_factor_controls.sbatch` (same four focused
  synthetic cells as the activity/error factor ablation; methods BP, DFA,
  DFA+norm, activity/error/K-nDFA, and activity/error/K-nDFA+norm; 5 seeds; 100
  epochs by default).
- Aggregate after the array finishes: `sbatch slurm/infodfa_normmatch_factor_controls_aggregate.sbatch`,
  or directly:
  `python analysis/aggregate_infodfa_normmatch_factor_controls.py --input-root
  results/infodfa_normmatch_factor_controls_v1 --output-dir
  results/infodfa_normmatch_factor_controls_aggregate_v1`.

## Adam / diagonal approximation control (paper uses activity-side rows only)
- Purpose: test whether conditioned-DFA gains reduce to an Adam-like diagonal
  second-moment learning-rate effect. The comparison is BP, DFA, hidden-weight
  Adam-DFA, diagonal square-root activity/error/K conditioning, activity/error
  nDFA, and full K-nDFA on the four focused synthetic cells.
- Run: `sbatch slurm/infodfa_adam_diagk_approx.sbatch` (GPU array;
  5 data seeds x 2 feedback seeds; damping grid {0.03,0.1,0.3,1.0}; Adam hidden
  lr grid {0.001,0.003}; hidden 256-128; 100 epochs).
- Aggregate: `sbatch slurm/infodfa_adam_diagk_approx_aggregate.sbatch`, or
  directly:
  `python analysis/aggregate_infodfa_adam_diagk_approx.py --inputs
  'results/infodfa_adam_diagk_approx_v1/shards/*/infodfa_adam_diagk_results.csv'
  --output-dir results/infodfa_adam_diagk_aggregate_v1`.
- Main outputs: `infodfa_adam_diagk_best_endpoints.csv`,
  `infodfa_adam_diagk_approximation.csv`, `infodfa_adam_diagk_curves.csv`,
  `infodfa_adam_diagk_learning_best.{pdf,png,svg}`, and
  `infodfa_adam_diagk_diagnostics.{pdf,png,svg}`.

## Decorrelation baseline (§4.3, "beyond decorrelation")
- Run: `sbatch slurm/infodfa_actwhiten_synthetic.sbatch` (128-cell grid, same recipe
  as the main sweep; methods dfa_random/dfa_actwhiten/ndfa_random/ndfa_random_kronecker).
  `dfa_actwhiten` preconditions the DFA update by (C+lambda I)^{-1/2} (ZCA decorrelation,
  power 1/2) vs nDFA's full inverse (C+lambda I)^{-1} (power 1).
- Aggregate: `analysis/aggregate_actwhiten.py`. Result: decorrelation recovers most of
  the input-side gain (+16..+34pp over DFA); the full inverse adds +5.8/+3.7/+0.8pp on
  nuisance/low-sample/mixed and over-conditions the clean control (-0.7pp).

## Tuned-BP control (§4.3)
- Run: `sbatch slurm/infodfa_bp_tuning_synthetic.sbatch` (BP only, lr grid
  {0.02,0.04,0.08,0.16,0.32} at every cell).
- Aggregate: `analysis/aggregate_bp_tuning.py` (tuned BP per cell via LOSO over lr;
  matched cell-by-seed LOSO robustness test vs the selected conditioned rule).

## BP-preconditioning control, matched learning rate (§4.3, Table 14)
- BP+precond data: `results/infodfa_bpwhiten_synthetic_v1` (same lr grid as Tuned-BP).
- The fair comparison reports BOTH BP and BP+precond at their best-of-grid lr per
  regime (NOT best-of-grid BP+precond vs fixed-lr BP). Per-regime best-lr means
  (x100): tuned BP = {nuisance 27.9, mixed 43.8, low-sample 53.3, task-aligned 92.0};
  BP+precond = {46.2, 44.6, 60.4, 91.9}; matched delta = {+18.3, +0.8, +7.1, -0.1}.
  On nuisance, plain BP did not exceed 27.9% on the tested five-rate grid
  (peaks at 0.08, falls higher),
  so the +18.3 is a regime lr-tuning cannot reach; on mixed the gain is mostly an
  lr effect tuned BP recovers.

## Historical KFAC-DFA control (invalid for a corrected two-sided claim)
- Run: `sbatch slurm/infodfa_kfac_control_synthetic.sbatch` (adds
  `ndfa_random_kronecker_bp`: K-nDFA with a BP-error left factor instead of DFA-error).
- Aggregate: `analysis/aggregate_kfac_control.py` (K-nDFA minus KFAC-DFA(bp) = +0.01pp mean).
- These archived values remain excluded. The corrected Fashion-MNIST
  `kndfa_bp` audits above supersede them as a source-specific negative control;
  neither is used as evidence of covariance-source equivalence.

## Local-rule baselines (sign-symmetry, Appendix F)
- Run: `sbatch slurm/infodfa_localrule_baselines_synthetic.sbatch` (adds `fa_sign`).

## Vision MLPs and ColoredMNIST (Tables 11, 17)
- Vision noisy-label sweep: `experiments/run_dfa_nmnc_comparison.py` (slurm `infodfa_nmnc_comparison.sbatch`).
- ColoredMNIST: `experiments/run_dfa_coloredmnist.py` (slurm `infodfa_coloredmnist.sbatch`).

## Regularization and norm-matching controls (activity-side rows support the paper)
- Run: `sbatch slurm/infodfa_controls.sbatch` (BP/BP+L2/BP+label-smoothing/BP+early-stop/
  DFA/DFA+norm-match/nDFA plus a historical two-sided row at one nuisance cell
  and one Fashion-MNIST cell, 5 seeds).

## Capable-model preconditioning vs norm-match (§4.3)
- Run: `sbatch slurm/infodfa_capable_normmatch.sbatch` (CIFAR-100 convnet, methods
  bp/local_loss/dfa_random/dfa_random_normmatch/ndfa_random/ndfa_random_kronecker;
  Adam; per-method lr {bp,dfa,normmatch 3e-4; ndfa 1e-4}; channels 64-128-256; 40 epochs;
  feedback-scale 0.3; 5 seeds x 5 feedback-seeds).
- Aggregate: `analysis/aggregate_capable_normmatch.py` (nDFA 23.6 vs DFA+norm-match 13.1).

## Spatial-Kronecker conv conditioning (Appendix H)
- Rule: `infogeo/conv_dfa.py:spatial_kronecker_conv_gradients` (method
  `ndfa_spatial_kron`) = channel-only nDFA plus a kernel-patch spatial second-moment
  factor (`kernel_spatial_covariance` over the kH*kW receptive-field positions,
  `solve_kernel_spatial`). The channel factor is identical to nDFA, so the
  difference isolates the spatial term.
- Clean test (drops into the capable CIFAR-10 dir): `sbatch
  slurm/infodfa_capable_cifar10_spatialkron.sbatch`; aggregate with
  `analysis/aggregate_spatial_kron.py` (spatial-Kron 64.6 vs channel-only nDFA 65.4
  = -0.78pp; the crossed-seed interval and Wilcoxon value are descriptive and
  do not establish a signed theoretical prediction).
- Both-directions nuisance sweep: `sbatch slurm/infodfa_spatialkron_nuisance.sbatch`
  (adds a class-independent low-frequency spatial nuisance via `--spatial-nuisance`
  alpha {0.5,1.0,2.0}, scale k=4, seeded by image index; methods bp/dfa_random/
  ndfa_random/ndfa_spatial_kron; same 40-epoch capable recipe). Aggregate with
  `analysis/aggregate_spatialkron_sweep.py`, which reports D(alpha)=spatial-Kron
  minus channel-nDFA (paired Wilcoxon + bootstrap CI) and the channel-nDFA-minus-DFA
  confound control. Because a channel-shared field still changes uncentered
  channel moments, the comparison does not fully isolate the spatial factor.
- Generality check on a second dataset (CIFAR-100 convnet): `sbatch
  slurm/infodfa_spatialkron_nuisance_cifar100.sbatch` (self-contained matched sweep,
  alpha {0,0.5,1.0,2.0} x {bp,dfa_random,ndfa_random,ndfa_spatial_kron}, same capable
  recipe, n-classes 100). Aggregate with `python analysis/aggregate_spatialkron_sweep.py
  results/infodfa_spatialkron_nuisance_cifar100`. Result: D rises monotonically with
  alpha as on CIFAR-10 (+2.3 -> +5.8pp), a descriptive replication of the trend;
  the clean-data sign differs (positive on CIFAR-100, where channel-nDFA is
  7.9pp below BP and within-patch anisotropy is still exploitable, vs the small clean
  loss on CIFAR-10 where channel-nDFA already ties BP).
- Robustness controls (damping + spatial scale) at alpha=1.0: `sbatch
  slurm/infodfa_spatialkron_controls.sbatch` (lambda in {0.1,1.0}; k=8); aggregate
  with `analysis/aggregate_spatialkron_controls.py`. Result: D is +1.4/+7.4pp at
  lambda 0.3/1.0 but reverses to -1.4pp at damping 0.1, and persists at k=8
  (+2.0pp). This sensitivity is compatible with small-direction amplification
  but does not identify it. sign(D) is
  jointly controlled by nuisance amplitude and damping; the both-directions sweep is
  the fixed-lambda=0.3 slice.

## CIFAR-100 convnet (Table 18)
- Run: `sbatch slurm/infodfa_hard_cifar100_confirm.sbatch`.

## ImageNet-100 substitution-depth boundary (Table 19)
- Block-output ZCA diagnostic: `sbatch slurm/infodfa_imagenet_strongform.sbatch` (13 configs:
  BP + {raw block-DFA, diagonal block-output, full block-output inverse-square-root conditioner} x 4 depths
  {layer4, layer3+4, layer2+3+4, all}; pretrained ResNet-18; `--dfa-norm unit`
  (no BP-norm oracle); `--whiten-mode {diag,full}`; 90 epochs; lr 0.1; image 176).
- Multi-seed extension (error bars on the cliff): `sbatch
  slurm/infodfa_imagenet_strongform_seeds.sbatch` (seeds 1-2 for all 13 configs;
  seed 0 is the original run; full-cov at layer2+3+4/all uses lr 0.01 to match the
  headline table). Outputs land in `results/imagenet100_strongform_v1/<tag>_seed<S>`.
- Deep-depth lr check for full-cov: `sbatch slurm/infodfa_imagenet_sf_lrcheck.sbatch`
  (full-cov at layer2+3+4 and all, lr {0.01,0.03}).
- Aggregate: `analysis/aggregate_imagenet_strongform.py` (also writes the
  three-seed mean±sem table the paper cites to
  `results/imagenet100_strongform_v1/strongform_multiseed_summary.csv`; for the
  two deep full-cov configs seed 0 is the lr-0.01 lr-check run, since the
  lr-0.1 originals collapse).

## Theory figure (Fig 1 for §2.1)
- Self-contained (no external data): the `_theory_sim` / `make_theory_conditioning`
  routines in the paper-source figure-assembly script. Panel C uses
  `task_in_high_var` to place the target in low- vs high-variance eigendirections.

## Additional appendix figures
- Feedback-variance collapse: `python analysis/compute_feedback_variance_collapse.py`
  (requires the noise-sweep and Mixer shards) writes
  `results/infodfa_feedback_variance_v1/`.
- Mode-timing theory validation: `python analysis/validate_mode_timing.py`
  (self-contained, CPU-only) writes `results/infodfa_mode_timing_v1/`.

## External DFA-Stall diagnostic
- The DFA-Stall reference implementation is vendored under
  `external/DFA-Stall/`; see `external/DFA-Stall/VENDORED_INFO.md` for
  provenance. No clone step is needed.
- Faithful comparison on the DFA-Stall MNIST/tanh setup:
  `python experiments/run_dfa_stall_comparison.py --output-dir
  results/dfa_stall_comparison_3seed_v1 --total-steps 1000 --hidden 300
  --batch-size 128 --lr 1e-3 --damping 0.3 --probe-n 1024 --eval-every 50
  --metric-every 10 --methods dfa ndfa kndfa --seeds 42 43 44 --device cuda`.
- Norm-matched diagnostic: same command with `--norm-match-hidden` and
  `--output-dir results/dfa_stall_comparison_normmatch_3seed_v1`.
- Aggregate both runs: `python analysis/aggregate_dfa_stall_comparison.py`, which
  writes `results/dfa_stall_comparison_overview_v1`.
- Error-side ablation: run `experiments/run_dfa_stall_comparison.py` with methods
  `dfa ndfa endfa kndfa` over damping `{0.03,0.1,0.3,1,3,10}` into
  `results/dfa_stall_error_ablation_damping_v1/{faithful,normmatch}_d*`, using
  `--norm-match-hidden` for the norm-matched condition. Aggregate with
  `python analysis/aggregate_dfa_stall_error_ablation.py`.

## Regenerating all figures

The composite paper figures are assembled by a script shipped with the paper
source (arXiv submission); it reads the aggregate roots listed above via
`INFODFA_RESULTS` (and optionally `INFODFA_LEGACY_RESULTS`) and is
headless-safe (Agg backend). Clean regeneration additionally requires the large
synthetic trajectory aggregate
`infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_all.csv`
(approximately 544 MiB, regenerated by the synthetic stress-suite commands
above) staged under either results root.

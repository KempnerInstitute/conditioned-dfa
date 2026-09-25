# Parallel baseline and mechanism studies, 25 September 2026

The user authorized the remaining justified experiments and moving pending work to the newly available `kempner_eng` partition. This launch adds **466 training runs in 209 array tasks**, plus nine dependent CPU analyses. A training run is one configuration and one global seed; an array task runs its declared seeds sequentially. Verification tasks are subsets of these counts, not additional experiments.

## Submission ledger

| Study | Configurations | Training runs | GPU array(s) | Partition / accelerator |
|---|---:|---:|---|---|
| CIFAR baseline tuning extensions | 38 | 76 | 48598245 | requeue / H100 |
| CIFAR baseline longer horizons | 9 | 18 | 48598247 | requeue / H100 |
| CIFAR FOOF tuning extensions | 18 | 36 | 48598252 | requeue / H100 |
| CIFAR FOOF longer horizons | 2 | 4 | 48598258 | requeue / H100 |
| MNIST/Fashion stronger baselines | 72 | 144 | 48599548, 48599726 | eng / H200 |
| Synthetic alignment interventions | 16 | 64 | 48599553, 48599731 | eng / H200 |
| CIFAR alignment interventions | 8 | 32 | 48599570, 48599734 | requeue / H100 |
| CIFAR mean/covariance controls | 34 | 68 | 48601403, 48601458 | requeue / H100 |
| Remaining synthetic damping boundaries | 12 | 24 | 48602047 | eng / H200 |

Every new requeue array has throttle 60. Eng arrays use throttle 12, subject to site limits. The digit and synthetic alignment arrays were moved while wholly pending; job IDs, dependencies and frozen scientific configurations were preserved. The two final pending tasks 48593949_47 and 48593949_60 from the previously submitted synthetic geometry study were also moved to eng, retaining H200 hardware. These four older training runs are outside the new 466-run count.

Analyses: 48601522, 48601530, 48601532, 48601538, 48601543, 48601548, 48601555, 48601557 and 48602049, in table order. They depend on termination of every array belonging to their study, audit provenance/completion, and retain missing and numerical-failure records. They cannot promote an incomplete grid into a completed selection. New runners' full arrays additionally depend on successful GPU verification of complete expected updates, paired seeds, finite endpoints and hashes. This technical gate uses no accuracy threshold.

## Baseline tuning and horizon checks

The completed initial CIFAR grid has 206 finite completions and ten numerical failures; its EMA FOOF companion has 46 completions and two numerical failures. All records remain in the analysis. Six baseline families and both FOOF families initially selected a boundary. Each receives the same outward factors of three and ten, crossed with adjacent existing values, excluding previously evaluated points. Select over the union of old and new candidates using mean final validation loss, then accuracy and unique case ID for ties. Failed candidates are ineligible, not replaced.

The nine baseline and two FOOF first-grid winners are also rerun for 400 rather than 200 epochs, from the paired initializations with stretched cosine schedules. These are provisional horizon-sensitivity controls, not continuation checkpoints, convergence guarantees, final confirmation, or matched-compute comparisons. CIFAR retains the original 45k/5k split, paired development seeds, preprocessing and H100 hardware.

All 340 runs in the first synthetic boundary extension and all 136 synthetic geometry runs completed and passed local provenance audits during this launch. Two task-aligned activity families still select damping boundaries after pooling the original and extension grids: BP without BN and DFA with BN. A second bounded application of the identical outward rule adds six configurations per family, with the same two development seeds and 500-epoch horizon. It retains all 554 earlier candidates and does not automatically continue to a third expansion. All 64 earlier synthetic horizon runs also completed. Geometry searches answer a different question; their damping flags are not claims of final optimization.

## Stronger digit baselines

Three settings: MNIST with tanh, FashionMNIST with tanh, and MNIST with ReLU. Compare BP, raw DFA, activity DFA and EMA FOOF-BP with six configurations each and two new development seeds (928101–928102). Tanh networks have three 300-unit hidden layers; ReLU networks have hidden widths 256 and 128. Train 60 epochs, or 25,800 updates, with batch size 128. Use only `train=True` data, a fixed 55k/5k split (seed 928005), pixel scaling by 255 and no augmentation. Official test data are never loaded.

For BP, DFA and activity DFA, compare SGD with momentum at learning rates 0.001, 0.01 and 0.1, and AdamW at 0.0001, 0.001 and 0.01. Activity damping is initially fixed at 0.3: this grid screens optimizer and horizon adequacy, not all activity/error damping choices. EMA FOOF uses SGD rates 0.03, 0.3 and 3 crossed with damping 0.3 and 3; normalized EMA 0.95, 50 calibration batches and inverse refresh every 100 updates. No weight decay is applied in this digit study.

Tanh uses stable one-versus-rest sigmoid BCE, summed over classes and averaged over examples. The manual BP calculation matches autograd. This intentionally replaces the historical runner's clamped-probability loss and is a stronger-baseline experiment, not a bitwise historical rerun. It does not yet establish E/K effects under the stronger recipes.

## Alignment interventions

At first-grid activity-selected optimizer/rate/damping settings, compare conditioning on every update, no updates, the first quarter, and the last quarter. Early and late interventions use exactly the same number of conditioned updates; the optimizer, learning-rate schedule and norm-matching rule are held fixed. Synthetic uses nuisance-dominant and task-aligned cells, each with/without BN, for 500 epochs. CIFAR uses both normalization conditions for 200 epochs. Four paired global seeds (927101–927104) are new to this stage. They vary initialization, minibatch order and feedback together; they do not isolate feedback-matrix variance.

At the same model states, measure raw DFA and conditioned gradient alignment to BP separately for every hidden layer. Use a fixed training probe, independent probe/augmentation RNGs, and restore all BN statistics and forward caches. Tests verify that probing leaves training unchanged and that checkpoint resume preserves trajectories and diagnostic histories. BP is an offline diagnostic and never enters the learner. Projection measurements concern gradient-like matrices before optimizer state, momentum or learning-rate scaling. Diagnostic work is timed separately; these instrumented runs are not practical-efficiency benchmarks.

Prespecified summaries: the first of three consecutive strictly positive raw-cosine observations defines observed alignment onset; non-onset is censored at the last observed update. Sparse probing limits temporal resolution. Integrate the sampled negative raw descent projection over updates 0–256 by the trapezoidal rule. Missing windows produce null values, not zeros. Report paired always-minus-never and early-minus-late contrasts, seed values and censoring. Four-seed outcomes are descriptive; do not add seeds based on significance.

## CIFAR mean/covariance intervention

For each normalization condition, retain the first-grid activity-selected optimizer and rate. Compare full uncentered moments, their diagonal, centered covariance, diagonal covariance plus the full mean outer product, and trace-matched isotropic covariance plus the mean outer product. Each receives three relative damping values: one-third, one and three times the anchor. Keep the raw numerator and forward network unchanged. All five use feature-space dense solves and norm matching; biases and BN-affine updates remain raw. Add matched no-conditioning and original sample-space activity references. Two original development seeds yield 68 runs.

A fixed augmented training probe records mean-energy fraction, centered participation rank and centered off-diagonal covariance energy, restoring BN state afterward. These are minibatch descriptions, not population statistics. Damping is relative to uncentered mean-square activity for all five operators, so centering changes the matrix without silently changing the damping reference. This is a balanced, provisional mechanistic comparison; dense inversion is deliberately shared and is not an efficiency claim. Three-hour allocations accommodate the 3,072-dimensional input-side solves.

## What remains conditional

The launch covers the currently defined baseline and first mechanism studies. Final held-out confirmation, final matched-work comparisons, feedback-only crossed replications, checkpoint swap interventions, error-anisotropy development and a larger neural-rendering/Transformer application are **not submitted**. Their settings or implementation are not yet fixed. They require the baseline/geometry/alignment review specified in the integrated plan; submitting arbitrary configurations now would not answer the intended questions. No test set is opened by any job in this ledger.

The appropriate next action is to inspect pooled selections, complete loss curves, remaining boundaries, censoring and per-seed mechanism outcomes; freeze the next scientific protocol from that evidence. Do not interpret successful GPU execution as confirmation of the proposed mechanism.

Code checks: 32 targeted tests pass for manual gradients, geometry algebra, probe neutrality, schedule work matching, resume identity, boundary extension, pooling, censoring and projection integration. Frozen configurations, source hashes, submissions, partition migrations and per-attempt hardware records live under `results/ndfa_strengthening_20260925/`. No manuscript claims or public submission were updated by this launch.

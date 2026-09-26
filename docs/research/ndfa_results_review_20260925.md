# nDFA development results: completed-job review

Checked 25 September 2026, evening Eastern time (26 September UTC). All 15 submitted strengthening studies have finished. The scientific result is encouraging for activity geometry and temporary conditioning, while stronger baselines limit broader performance claims. These are validation-development results, not held-out confirmation.

## Execution and reproducibility

Across the full strengthening campaign, 1,998 runs finished normally and 40 ended in recorded numerical failure, accounting for all 2,038 declared runs. In the latest 466-run round, 451 finished normally and 15 failed numerically. There are no missing runs. All 15 scheduled analysis jobs completed. The 32 canceled entries in scheduler history are earlier pending tasks migrated between partitions; their replacements produced the expected results. No recovery run is needed.

| Latest study | Declared runs | Successful | Numerical failures |
| --- | --- | --- | --- |
| cifar_baseline_boundaries_v1 | 76 | 72 | 4 |
| cifar_baseline_horizon_v1 | 18 | 18 | 0 |
| cifar_foof_boundaries_v1 | 36 | 25 | 11 |
| cifar_foof_horizon_v1 | 4 | 4 | 0 |
| digits_v1 | 144 | 144 | 0 |
| alignment_synthetic_v1 | 64 | 64 | 0 |
| alignment_cifar_v1 | 32 | 32 | 0 |
| geometry_cifar_v1 | 68 | 68 | 0 |
| synthetic_boundaries_v2 | 24 | 24 | 0 |

The latest failures are four forward-decorrelation runs and eleven FOOF runs at aggressive extension settings with large learning rates or inadequate damping. They remain in the development record and do not enter selection as successful trials. None of the digit, alignment, geometry or longer-horizon runs failed numerically.

Recomputed all 15 summaries from the saved endpoints and histories after checking frozen source hashes, configurations, manifests, completed update counts and final-checkpoint hashes. Every candidate, selected configuration and endpoint count agrees exactly with the scheduled analysis output. Within each study, paired conditions share data, initial weights, feedback matrices and complete minibatch/augmentation streams. This is a saved-artifact audit; it does not rerun training or independently regenerate every validation prediction. No official test data were opened.

## CIFAR: activity conditioning survives stronger baselines

The following means use two paired development seeds at 200 epochs. Select each method by mean final validation cross-entropy over its pooled original and extension grids. Accuracy is displayed at that selected configuration; it is not a separate accuracy-based selection. All eleven CIFAR families now select interior points in the tested coordinates.

| Method family | Validation accuracy | Validation CE |
| --- | --- | --- |
| bp_none_none | 63.06% | 1.067 |
| bp_none_activity | 66.07% | 1.003 |
| bp_bn_none | 66.60% | 0.982 |
| bp_bn_activity | 66.84% | 0.956 |
| dfa_none_none | 61.19% | 1.132 |
| dfa_none_activity | 63.86% | 1.065 |
| dfa_bn_none | 64.56% | 1.026 |
| dfa_bn_activity | 67.71% | 0.952 |
| dfa_fd_none | 63.48% | 1.063 |
| bp_none_foof | 72.18% | 0.805 |
| bp_bn_foof | 72.47% | 0.910 |

Activity DFA improves on raw DFA by 2.67 percentage points without BN and 3.15 points with BN; both paired seeds favor it in each comparison, and CE also improves. Its BN result slightly exceeds ordinary BP+BN under the tested recipes. The strongest baseline is EMA FOOF-BP, which reaches about 72% accuracy and remains substantially ahead. These are equal-epoch development comparisons, not evidence that nDFA beats BP at matched compute.

At 400 epochs, first-grid activity-DFA+BN reaches 68.46%, compared with 65.51% for raw DFA+BN; FD-DFA reaches 66.00%. Thus, the within-BN activity benefit does not disappear merely by doubling this horizon. Several methods gain accuracy while validation CE worsens, and FD-DFA continues to improve. A fixed horizon is not proof of convergence. The 400-epoch runs use fixed first-grid winners with stretched schedules, not the newly pooled winners, so differences between the 200- and 400-epoch tables are not always pure horizon effects.

## Geometry: covariance matters beyond the tested mean controls

The CIFAR mechanism study holds optimizer and learning rate at first-grid activity anchors, changes only the moment operator, and searches the same three damping values for each operator. Its two seeds are development seeds. All five operators use the same dense-solve implementation; separately selected endpoint means follow.

| Moment operator | Without BN | With BN |
| --- | --- | --- |
| full | 64.76% | 68.55% |
| centered | 64.04% | 67.91% |
| diagonal | 56.45% | 64.21% |
| diagonal_covariance_plus_mean | 58.67% | 63.10% |
| isotropic_covariance_plus_mean | 58.01% | 63.55% |

Full and centered covariance outperform the tested diagonal and mean-preserving alternatives. At each matched damping, full moments beat every diagonal/mean-only control in both seeds and both normalization conditions. Those six comparisons per normalization reuse two seeds and are not six independent replications. Centered covariance is close to full moments on CIFAR; with BN it has slightly lower selected CE despite lower accuracy. On nuisance-dominant synthetic data without BN, centered covariance exceeds full moments at every tested damping/seed pair. On task-aligned synthetic data, gains disappear and BN can favor the simpler operators.

This is evidence against explaining the benefit solely by suppressing the mean direction under these recipes. It supports a role for centered covariance structure, with scope limited by provisional optimizer/rate anchors, two seeds, and remaining damping boundaries in several control families. The dense full rule and its mathematically equivalent sample-space bridge differ by about 0.6 percentage points in the CIFAR BN trajectory at the anchor; implementation sensitivity should be retained in confirmation rather than treating those trajectories as interchangeable.

## Temporary conditioning: the most useful new intervention

Four paired global seeds compare always, never, first-quarter and last-quarter conditioning at the same activity-selected optimizer/rate. These are fixed-recipe interventions; the never-conditioned arm is not the independently tuned raw-DFA baseline.

| Condition | Never | Always | First quarter | Last quarter |
| --- | --- | --- | --- | --- |
| cifar10, none | 55.29% | 63.74% | 63.52% | 55.42% |
| cifar10, bn | 64.24% | 68.12% | 67.93% | 64.10% |
| nuisance, none | 21.46% | 37.57% | 37.15% | 21.45% |
| nuisance, bn | 29.22% | 42.43% | 43.08% | 29.28% |

First-quarter conditioning retains nearly all of the full-training accuracy benefit. Early minus late is positive in all four CIFAR seeds with and without BN, and in all four nuisance seeds in both normalization conditions. On CIFAR with BN, first-quarter conditioning averages 67.93% versus 68.13% for always conditioning, with roughly one-third less measured learning work in this instrumented study. It reduces the number of conditioned updates by 75%; that is not a 75% reduction in total training cost. This gives a concrete candidate for fresh-seed matched-work evaluation.

The mechanistic explanation is not settled. Raw alignment onset is often unchanged or later with conditioning; harmful projected-gradient area decreases in some layers and conditions but not all. Do not claim that conditioning universally removes an anti-alignment phase. Moreover, the first and last quarters have equal update counts but different learning rates: the nominal summed learning rate is about 15 times larger in the first quarter. This is a diagnostic of the schedule, not a measure of actual Adam or momentum displacement. Equal-learning-rate windows or checkpoint-branch interventions are needed to distinguish an early representation effect from schedule timing. A matched BP activity-conditioning intervention is also needed before calling this DFA-specific.

## Strong digit baselines narrow the older claim

| Setting | BP | DFA | Activity DFA | FOOF-BP |
| --- | --- | --- | --- | --- |
| mnist_tanh | 98.27% | 97.99% | 97.25% | 98.38% |
| fashion_mnist_tanh | 90.53% | 89.11% | 89.51% | 89.99% |
| mnist_relu | 97.86% | 97.16% | 97.32% | 97.83% |

With stronger optimizers and 60 epochs, the old large activity gains on digits largely disappear. Activity DFA is below raw DFA on MNIST-tanh, modestly above it on Fashion-tanh, and close on MNIST-ReLU. The Fashion accuracy increase accompanies worse validation BCE in both seeds. Nine of twelve families retain a tuning boundary and activity damping was fixed, so this is a baseline-adequacy screen rather than a final optimized ranking. It does establish that the historical short-budget improvements should not be presented as general superiority on these datasets. E/K were not tested in this stronger-baseline grid.

## Synthetic regimes and the next scientific decisions

| Regime | DFA | Activity DFA | DFA+BN | Activity DFA+BN |
| --- | --- | --- | --- | --- |
| nuisance | 15.99% | 35.83% | 17.02% | 37.10% |
| low_sample | 20.78% | 23.75% | 23.93% | 28.45% |
| mixed | 33.15% | 38.12% | 38.39% | 40.33% |
| task_aligned | 96.96% | 96.88% | 96.23% | 96.30% |

The pooled synthetic search preserves the strong nuisance-regime benefit, with smaller gains in low-sample and mixed regimes and essentially no gain in the task-aligned regime. Selected raw DFA now has ordinary finite training losses: roughly 1.4 in the nuisance cell, rather than the extreme losses in the historical shared-rate sweep. The benefit therefore survives removing that earlier catastrophic instability. Conditioning also lowers training loss substantially, so this comparison still includes an optimization benefit and does not isolate generalization at matched training fit. The two remaining synthetic tuning boundaries are task-aligned activity families selecting progressively weaker conditioning. This is not a reason to keep launching unbounded searches.

Prioritize the covariance mechanism and temporary conditioning. Next, freeze a small fresh-seed CIFAR comparison of tuned DFA+BN, full-time and early-only activity DFA+BN, BP+BN, and FOOF-BP at matched measured work, with validation-selected horizons/checkpoints. Resolve the relevant geometry-control damping boundaries, then confirm full/centered/mean-preserving contrasts. Use an equal-rate early/late or shared-checkpoint intervention, including a BP control, to test the mechanism. Keep error-factor expansion and the larger application conditional on that result. The results justify a focused stronger paper; they do not establish a general advantage over strong BP baselines.

No training jobs were resubmitted during this check: every declared run already has an audited terminal record. Raw evidence, recomputed summaries, paired values and failure causes are in `results/ndfa_strengthening_20260925/job_results_check_v1/`.

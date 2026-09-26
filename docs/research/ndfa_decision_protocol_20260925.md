# Frozen decision round for nDFA

**Authorization:** the user requested a full effort toward a prompt scientific decision after reviewing the completed development studies. This round is finite. Its purpose is to decide which claims justify a stronger paper and whether a larger application is worth pursuing. Its outcome is not an estimate of conference acceptance.

## Allocation and execution

| Study | Training runs | Role |
|---|---:|---|
| Fixed 200-epoch confirmation | 80 | Eight methods, ten paired new seeds |
| Matched-work development | 192 | Eight methods, six candidates, two budgets, two development seeds |
| Matched-work confirmation | 160 | Eight selected methods at two budgets, ten paired new seeds |
| Geometry damping extension | 40 | Five operators, four additional damping values, two development seeds |
| Geometry confirmation | 70 | Five selected operators and two reference rules, ten new seeds |
| Constant-rate timing intervention | 128 | BP/DFA, two shared rates, four policies, eight new seeds |
| GPU execution checks | 6 | Technical checks; excluded from scientific sample counts |

The maximum is 670 scientific training runs plus six technical runs. One Slurm task runs one seed/configuration. No task waits for another seed in the same configuration. The two development stages automatically freeze and launch their confirmation cohorts. Evaluations wait for both development selections to be frozen. The final report waits for all four confirmation/intervention evaluations.

H100 comparisons may run in `kempner_h100_priority` or `kempner_requeue`, retaining the same accelerator model. The constant-rate intervention uses H200 throughout, with `kempner_eng` available if free and `kempner_requeue` available in parallel. Requeue arrays use throttle 60, subject to actual site allocation. Hardware, attempts and source/configuration hashes are saved. This removes avoidable serial work but does not promise a scheduler start time.

Based on the preceding runs, an ordinary configuration takes a few minutes per seed, and a dense geometry configuration about ten minutes. The full round therefore requires tens of GPU-hours and can finish in hours with enough simultaneous allocations. The operational target is a first decision within a day, conditional on queue availability and successful technical checks.

## Shared scientific and data protocol

Retain the CIFAR-10 45k/5k training/validation identities (split seed 925005), training-only channel normalization, reflected random crops and horizontal flips, 1024/512 ReLU hidden widths, batch size 256, feedback scale 0.1, float32 training and float64 evaluation loss. TF32 remains disabled. Learning uses manual BP/DFA updates; the local learner receives no BP diagnostic signal. Bias and BN-affine updates follow the existing implementations.

The practical comparison includes tuned DFA+BN, full-time activity DFA+BN, early-only activity DFA+BN, BP+BN, component-matched BP+activity+BN, EMA FOOF-BP with and without BN, and forward-decorrelated DFA. FOOF retains normalized EMA 0.95, 50 calibration batches, inverse refresh every 100 updates, all-weight conditioning and raw bias/BN-affine updates. The component-matched activity rule retains hidden-weight norm matching.

The fixed-epoch study uses the latest pooled development winners. Its early-only rule shares the full-time rule's optimizer, rate and damping. Practical confirmation seeds are 930101–930110. Geometry uses 932101–932110, and the timing intervention uses 931101–931108. These global seeds jointly specify initialization, feedback and sampling; they are not separate feedback-only replicates. No additional seeds will be added according to significance.

Training never loads `train=False`. Test evaluation occurs in a separate gated stage after development selection and complete terminal training records. Both the final and validation-selected checkpoints are evaluated and their logits retained. The fixed CIFAR test set has been used earlier in this project: these are new-seed confirmations with prospective selection rules, not a claim that the project's test data were never previously inspected. No new test outcome selects a method, damping, horizon or seed.

## Fixed epochs and matched work

The fixed-epoch study trains 200 epochs with the existing five-epoch warmup and cosine schedule. First-quarter conditioning uses the first quarter of updates. Save the best validation-CE checkpoint at declared evaluation points (initialization, epoch 1, then every five epochs); the final checkpoint is also retained. The early-versus-full noninferiority comparison uses final fixed-epoch test accuracy, preventing checkpoint selection from obscuring the persistence question.

Matched-work budgets are 200 and 400 seconds of synchronized H100 learning work. The 200-second budget is primary. Charge minibatch sampling, augmentation, forward and credit computations, all conditioning and inverse updates, optimizer work, finite checks and FOOF calibration. Validation and artifact export are timed separately. Report the final single-update overshoot and attempts. Repeated hardware interruption requires inspection; discarded computation is not a claimed speedup.

For matched work, warmup occupies 2.5% of the work budget and cosine decay follows work progress. Early-only conditioning occupies the first quarter of measured learning work. This differs explicitly from the first-quarter-of-updates policy in the fixed-epoch study and receives its own validation development. All methods have the same validation opportunities at approximately 5% work increments, and the selected checkpoint is the lowest validation CE available within the completed budget, allowing only the documented final-update overshoot.

For full-time and early-only activity DFA, the six development candidates comprise two already documented activity settings (pooled and first-grid) crossed with learning-rate factors 0.5, 1 and 2. The remaining methods receive six learning-rate factors around their pooled winner: 1/3, 0.5, 1, 1.5, 2 and 3. Their operator settings stay fixed. Select by the mean validation-selected CE over seeds 925101–925102, then accuracy and case ID for ties. This is a bounded local refinement, not exhaustive optimization. Report any edge choice; do not automatically extend it.

## Covariance controls

Focus on the successful BN condition. Keep the first-grid activity optimizer and learning rate fixed at AdamW 0.001. All five operators use the same dense solve, raw numerator, norm matching and relative damping reference. Extend each operator's existing damping grid of 1/3, 1 and 3 with 1/30, 0.1, 9 and 30. The operators are full uncentered moment, centered covariance, diagonal uncentered moment, diagonal covariance plus the mean outer product, and trace-matched isotropic covariance plus the mean outer product.

Pool these results with the frozen earlier BN geometry candidates. Choose damping separately for each operator by mean final validation CE over the same two development seeds. Freeze ten-seed confirmation for the five selected operators, a matched raw update, and the original sample-space activity rule. The final 200-epoch checkpoint is primary for this mechanistic comparison. Dense and sample-space trajectories remain separate to reveal implementation sensitivity.

## Equal-rate timing and the BP control

Use BN, AdamW, relative activity damping 1, 200 epochs, and constant learning rates 0.0001 and 0.0003 shared by BP and DFA. Compare always, never, first-quarter and last-quarter conditioning at each rate. The first and last quarters contain equal numbers of updates at the same scalar learning rate. Optimizer state still evolves with the trajectory, which is part of the intervention. No warmup or decay is applied in this study.

The primary rate is 0.0003; 0.0001 is a prespecified robustness check. Test both the DFA early-minus-late effect and its paired difference from BP's early-minus-late effect. A DFA timing benefit alone cannot establish DFA specificity. This study addresses schedule timing; it does not by itself establish a circuit mechanism or isolate feedback-matrix variability.

## Decision criteria and uncertainty

Use paired model-seed contrasts, Student t intervals and all declared seeds. Numerical failures remain in the record. A contrast missing a declared successful pair is unavailable for the primary decision; do not report a selected successful subset as confirmation.

The primary family contains six two-sided paired tests, with Holm correction at 0.05:

1. Early activity DFA versus tuned DFA at the 200-second work budget.
2. Full-time activity DFA versus tuned DFA at that budget.
3. Full moment versus diagonal covariance plus mean in the geometry confirmation.
4. Centered covariance versus diagonal covariance plus mean.
5. DFA early versus late at constant learning rate 0.0003.
6. The DFA-minus-BP interaction for that timing contrast.

A practical benefit requires an adjusted significant positive contrast, a mean accuracy increase of at least one percentage point, and no increase in mean test CE. Full-time and early-only claims remain distinguishable. Report 400-second results as prespecified secondary evidence, even if their direction differs.

The temporary-conditioning efficiency criterion additionally requires the final fixed-epoch accuracy contrast's one-sided 95% lower bound to exceed -0.5 percentage points, and a mean paired learning-work ratio no greater than 0.8. State this separate noninferiority analysis explicitly; do not infer equivalence from a nonsignificant difference.

If practical, covariance and BP-interaction evidence all pass, advance the mechanism-focused paper and consider one demanding application. If practical and covariance evidence pass without DFA-specific timing, advance a narrower geometry/efficiency paper and limit the mechanism claim. Otherwise narrow the claims before spending resources on scale. Strong FOOF-BP and FD-DFA comparisons remain central regardless of outcome. No branch implies that the manuscript is ready for a specific venue.

The automated endpoint is `results/ndfa_strengthening_20260925/decision_round_v4/decision.md`, accompanied by paired values, adjusted tests, prediction archives and `decision.json`. The unsubmitted v1 and v2 snapshots are retained. Version 1 stalled in an NFS advisory lock; version 2 was rejected because the cluster excludes requeue from multi-partition submissions. Neither submitted any jobs. Version 3 was superseded before training began to give simultaneous audit writers unique temporary files; all ten pending scheduler entries were canceled. Version 4 uses atomic directory renames, unique temporary files, and single-partition submissions with the same scientific allocation and decision criteria. The manuscript is revised only after reviewing that evidence.

# Scorecard for pre-registered predictions

Predictions registered in PREDICTIONS.md (commit 533ef25, 2026-07-09 11:25,
before job completion; P5 in f5b9596). Scored 2026-07-09 afternoon from:
results/imagenet100_noisy_deconfound_v1 (21 runs, 91 epochs, 3 seeds),
results/infodfa_bn_baseline_v1 (128 synthetic cells + 12 Fashion cells),
results/infodfa_amortized_refresh_v1, results/benchmark_overhead_v1,
results/imagenet100_strongform_v1/ndfaFull_*_lr01_seed{1,2}.

Historical K-nDFA rows in P2--P5 predate the per-example error-moment
normalization correction. They remain part of the timestamped scorecard but
are not evidence for the corrected error-side or two-sided claims; P6 is the
first preregistered corrected two-sided entry.

## P1 — Noisy ImageNet-100 deconfound (40% label noise, final val top-1, n=3)

| config | layer4 | all-block |
|---|---|---|
| BP (depth-free) | 60.43 ± 0.18 (best-epoch 69.5) | — |
| raw block-DFA | 53.36 ± 0.51 | 35.57 ± 1.34 |
| diag channel-nDFA | 55.46 ± 0.16 | 42.85 ± 0.42 |
| full-cov ZCA | 39.71 ± 1.11 | 40.56 ± 0.28 |

- **P1a — CONFIRMED for the diagonal conditioner, REFUTED for full-cov at
  layer4.** Diag-nDFA flips the clean-protocol ordering at BOTH depths:
  layer4 +2.1 pp (clean: −0.5), all-block **+7.3 pp** (clean: −3.3). The
  conditioning advantage reappears at scale under noise, largest at the
  deepest substitution — the scale×regime confound resolves in favor of
  the regime explanation. Full-cov ZCA at lr 0.1 destabilizes at layer4
  under noise (39.7 vs 53.4) though it still beats raw DFA at all-block
  (+5.0); the strongest operator is learning-rate-fragile in the noisy
  regime, consistent with its clean-protocol LR sensitivity.
- **P1b — CONFIRMED.** Depth degradation persists for every method
  (diag: 55.5 → 42.9; raw: 53.4 → 35.6). Noise moves the
  conditioned-vs-raw delta, not the depth trend.
- **P1c — CONFIRMED** (final-epoch): BP-vs-best-local gap at layer4
  shrinks from 6.2 pp (clean) to 4.9 pp (noisy). Caveat: BP's final
  (60.4) sits 9.1 pp below its best epoch (69.5) — BP memorizes noisy
  labels late while all local rules end at ≈ their best (no late
  overfitting). Against best-epoch BP the gap is larger (14.1 pp); both
  framings should be reported.

## P2 — DFA+BatchNorm baseline (synthetic regime means over 32 cells; test acc)

| regime | DFA | DFA+BN | nDFA | K-nDFA | Δ(nDFA−DFA+BN) |
|---|---|---|---|---|---|
| nuisance-dominant | 13.4 | 47.4 | 53.3 | 56.8 | **+5.9 ± 0.9** |
| low-sample/noisy | 34.8 | 66.7 | 65.7 | 65.8 | −1.0 ± 0.3 |
| mixed-context | 25.3 | 50.7 | 47.3 | 50.0 | −3.4 ± 0.6 |
| task-aligned | 74.5 | 87.5 | 90.4 | 83.3 | +2.9 ± 0.5 |

(cell-matched, full-rank feedback, same data/feedback seeds; BN values from
new runs, others from the paper sweeps at matched cells.)

- **P2a — CONFIRMED.** DFA+BN improves raw DFA everywhere (+13 to +34 pp).
- **P2b — PARTIALLY (magnitude over-predicted).** nDFA beats DFA+BN in
  nuisance-dominant by +5.9 pp (K-nDFA +9.4 pp) — sign as predicted but
  below the registered ≥10 pp; and DFA+BN matches/exceeds nDFA in
  low-sample (−1.0) and mixed (−3.4). Structure: the nDFA advantage is
  concentrated in LOW-DATA nuisance cells (+10–16 pp at n_train 512–1024,
  ≈0 at 4096) — BN catches up given data; explicit conditioning wins when
  samples are scarce. BatchNorm is therefore an implicit partial
  conditioner for DFA, and the residual full-covariance gain survives
  where the law predicts it (nuisance-dominant), sharpest at low n.
- **P2c — REFUTED in the favorable direction.** On noisy Fashion-MNIST
  the gap is not uniformly small: nDFA beats DFA+BN by +6.1 pp mean over
  noisy cells, growing to +12–16 pp at n_train 10000.
- **P2d — REFUTED.** DFA+BN also improves the task-aligned control by
  +13 pp (BN rescues DFA's label-noise degradation there); BN is broadly
  helpful to DFA, not nuisance-specific. (BP+BN barely changes BP —
  +0.4 pp mean — so BN's large effect is specific to the local rule's
  activity statistics, itself consistent with the mechanism.)

## P3 — Amortized covariance refresh (final acc delta vs k=1)

- **P3a — PARTIALLY: CONFIRMED on synthetic, REFUTED on vision.**
  Synthetic nuisance cell: k=10 +0.5 pp, k=50 +1.6 pp (staleness is
  neutral-to-mildly-regularizing). Noisy Fashion-MNIST: nDFA k=10
  **−17.0 pp**, k=50 −13.1 pp — stale inverses badly hurt on real data.
  K-nDFA is staleness-robust on both (−3.5/+1.6 pp).
- **P3b — REFUTED as stated** (the vision deficit is not "a few pp" and
  not confined to early epochs). CONSEQUENCE: the paper may not claim
  amortized ~1.5–2× as accuracy-free; the claim must be scoped to the
  synthetic/stationary case or dropped.

## P4 — Overhead benchmark (per-step wall-clock ratio vs BP, H200 GPU)

- **P4a — CONFIRMED.** nDFA 3.8× (synthetic MLP) / 10.2× (vision MLP);
  K-nDFA 6.6× / 15.4×; K-nDFA > nDFA everywhere. Measured band brackets
  the paper's 4.7–18.8× development figures. CPU: 4.0–8.8× / 7.1–13.3×.
- **P4b — CONFIRMED (boundary).** k=10 refresh: 1.47× (synthetic) and
  2.17× mean / ≈1.2× median (vision; mean inflated by refresh-step
  spikes). But see P3: the speed is real, the accuracy cost on vision
  is not resolved.

## Missing-seeds completion (not a prediction; closes a disclosure)

ndfaFull lr 0.1 is now 3-seed: layer2+3+4 = 50.61 ± 0.26 (was single-seed
51.0), all-block = 39.32 ± 0.36 (was single-seed 38.6). The single-seed
values were representative; the paper's n=1 disclosures can be replaced
with 3-seed mean ± sem.

## P5 — MLP-Mixer, token/channel conditioning (CIFAR-10, final test acc, 5×3 seeds)

| cell | BP | DFA | nDFA | K-nDFA | Δ(nDFA−DFA) |
|---|---|---|---|---|---|
| clean, LayerNorm | 60.5 | 14.4 | 42.7 | 30.5 | +28.4 |
| noise 0.4, LayerNorm | 45.3 | 11.7 | 36.7 | 25.3 | **+25.0** |
| clean, no LN | 61.4 | 9.5 | 26.3 | 22.8 | +16.8 |
| noise 0.4, no LN | 44.7 | 9.5 | 16.6 | 15.8 | +7.1 |

- **P5a — CONFIRMED.** Conditioning survives the token/channel
  factorization: +25.0 pp over raw DFA under 40% noise in the standard
  (LayerNorm) Mixer.
- **P5b — CONFIRMED on sign, wrong on magnitude direction.** The gain with
  LayerNorm is positive and in fact LARGER than the CIFAR-10 MLP-tier gain
  (+14.9 pp), not smaller as registered.
- **P5c — REFUTED, informatively.** Ablating LayerNorm SHRINKS the gain
  (+25.0 → +7.1 pp under noise) instead of growing it: without LN the
  local rules barely train (raw DFA at chance), so LN is a trainability
  prerequisite that conditioning complements rather than a substitute
  that absorbs it. The registered failure mode ("LN provides the usable
  share") is excluded by the data: LN alone leaves DFA at 11.7% where
  LN+conditioning reaches 36.7%.
- The archived K-nDFA row trails nDFA in all four cells. Because it predates
  the error-moment correction, this is retained as a historical outcome rather
  than evidence against the corrected error factor.

## P6 — Clean Fashion-MNIST error-factor replication (final test, 5×3 seeds)

Protocol and pass/fail criteria were registered in `PREDICTIONS.md` at commit
`ef795e1`, before job 30989996 selected damping and before confirmation job
30991467 ran. Development independently selected $\lambda_A=0.03$ and
$\lambda_E=30$ on a fixed 5,000-example training-validation split. The error
optimum is interior to the eight-value grid; the activity optimum is at the
lower boundary. Confirmation froze the pair on five fresh model/data-order
seeds crossed with three feedback seeds, averaged feedback seeds within model
seed, and evaluated clean Fashion-MNIST test data only after the final update.

| method | test accuracy | test loss |
|---|---:|---:|
| DFA | 61.98 ± 0.17 | 1.938 ± 0.012 |
| activity nDFA | 80.29 ± 0.10 | 1.575 ± 0.007 |
| error nDFA | 63.74 ± 0.41 | 1.842 ± 0.010 |
| K-nDFA | **80.79 ± 0.09** | **1.500 ± 0.007** |
| K-nDFA, BP-error source | 80.29 ± 0.11 | 1.576 ± 0.007 |

- **P6a — CONFIRMED.** Error nDFA improves raw DFA by +1.767 pp, positive
  in all 5/5 model-seed units; test loss improves by 0.096.
- **P6b — CONFIRMED.** K-nDFA improves activity nDFA by +0.501 pp, positive
  in 5/5 units, and lowers test loss by 0.075.
- **P6c — NUMERICALLY SATISFIED, BUT THE COMPARATOR IS INCONCLUSIVE.** Local
  K-nDFA is +0.499 pp above the nonlocal BP-error-source comparator, positive
  in 5/5 units and just inside the registered absolute 0.5-pp margin. A
  post-run spectral audit shows that the frozen $\lambda_E=30$ is on the local
  DFA-error scale, whereas the BP-error covariance is much smaller. Its damped
  inverse is therefore nearly scalar, and layerwise norm matching removes that
  scalar: the BP-source update is effectively activity nDFA, consistent with
  their indistinguishable 80.29% endpoints. P6c cannot support a claim about a
  locality penalty or covariance-source equivalence. The registered absolute
  margin was also worded two-sided even though the failure interpretation
  concerned only a deficit of the local source.

  On the frozen seed-70/feedback-0 trajectory, sampled every ten steps,
  BP-error $\lambda_{\max}$ is 0.040--2.84 with trace 0.22--4.34, versus local
  DFA-error $\lambda_{\max}$ 8.59--505 and trace 42.9--723. After norm matching,
  the BP-source update cosine with activity nDFA is at least 0.99995; the
  local-source cosine reaches 0.833.

### Post-hoc scale-matched P6c addendum

A BP-source-only validation sweep over
`{0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30}` on the original
development split selected $\lambda_E^{BP}=3$. Dampings 3 and 10 tied at
82.90% mean validation accuracy; lower validation loss breaks the tie in favor
of 3. This selection is post hoc and is not part of the registered P6 result.

Fresh model/data-order seeds 80--84 crossed with feedback seeds 0--2 give:

| method | test accuracy | test loss |
|---|---:|---:|
| activity nDFA | 80.25 ± 0.07 | 1.5721 ± 0.0022 |
| local K-nDFA | **80.64 ± 0.05** | **1.4955 ± 0.0045** |
| K-nDFA, retuned BP-error source | 80.27 ± 0.07 | 1.5739 ± 0.0024 |

- Local K-nDFA minus activity nDFA: +0.391 ± 0.040 pp, positive in 5/5;
  loss −0.0765 ± 0.0028, lower in 5/5.
- Retuned BP source minus activity nDFA: +0.015 ± 0.011 pp, positive in 4/5;
  loss +0.0019 ± 0.0015, lower in 2/5.
- Retuned BP source minus local K-nDFA: −0.375 ± 0.042 pp, positive in 0/5;
  loss +0.0784 ± 0.0027, lower in 0/5.

The live, scale-matched comparison does not rescue a source-equivalence claim.
Instead, it supports a narrower source-specific conclusion in this setting:
conditioning by the local DFA-error second moment improves activity nDFA,
whereas conditioning by transported BP-error second moments does not. At the
selected damping 3, the BP factor's damped spectral ratio reaches 1.95, but the
norm-matched update cosine with activity nDFA remains at least 0.9984 on the
audited trajectory.

Consequence at P6: the error-side and incremental two-sided signs replicate on
two clean datasets, but both use the same architecture and one-vs-rest loss.
P7 below addresses that architecture/loss limitation. The source swap should be
reported as the post-hoc source-specific negative control above, not as
evidence of equivalence or absence of a locality penalty.

## P7 — ReLU/softmax architectural replication (final test, 8×3 seeds)

The validation-only development protocol used a 256--128 ReLU MLP with
multiclass softmax cross-entropy, 1,000 updates, and layerwise hidden-gradient
norm matching. Seeds 0--2 independently selected `lambda_A=3` and
`lambda_E=0.1`; the protocol and pass/fail criteria were committed before test
evaluation. Confirmation used fresh model/data-order seeds 100--107 crossed
with feedback seeds 0--2 and averaged feedback seeds within model seed.

| method | test accuracy | test loss |
|---|---:|---:|
| DFA | 87.08 ± 0.79 | 187.339 ± 95.506 |
| activity nDFA | 95.81 ± 0.03 | 0.1385 ± 0.0005 |
| error nDFA | 94.61 ± 0.16 | 1.242 ± 0.187 |
| K-nDFA | **96.71 ± 0.04** | **0.1103 ± 0.0009** |

- **P7a — CONFIRMED.** Error nDFA improves DFA by +7.526 ± 0.860 pp,
  positive in 8/8 model-seed pairs; test loss is also lower in 8/8.
- **P7b — CONFIRMED.** K-nDFA improves activity nDFA by +0.900 ± 0.042 pp,
  positive in 8/8, and lowers loss by 0.0282 ± 0.0011 in 8/8. Both exact
  two-sided Wilcoxon tests give p=0.0078125.
- **P7c — CONFIRMED WITH A SCOPE LIMIT.** Both factor-specific signs transfer
  from a three-hidden-layer tanh/one-vs-rest model to a two-hidden-layer
  ReLU/softmax model on fresh MNIST seeds. Dataset choice was made after an
  exploratory ReLU Fashion-MNIST development pilot failed to retain the
  incremental K-nDFA gain under independent factor selection. This is therefore
  architecture/loss replication on MNIST, not ReLU dataset generality.

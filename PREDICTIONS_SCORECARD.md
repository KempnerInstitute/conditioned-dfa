# Scorecard for pre-registered predictions

Predictions registered in PREDICTIONS.md (commit 533ef25, 2026-07-09 11:25,
before job completion; P5 in f5b9596). Scored 2026-07-09 afternoon from:
results/imagenet100_noisy_deconfound_v1 (21 runs, 91 epochs, 3 seeds),
results/infodfa_bn_baseline_v1 (128 synthetic cells + 12 Fashion cells),
results/infodfa_amortized_refresh_v1, results/benchmark_overhead_v1,
results/imagenet100_strongform_v1/ndfaFull_*_lr01_seed{1,2}.

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
- K-nDFA trails nDFA in all four cells — the error-side factor again
  fails to earn its complexity.

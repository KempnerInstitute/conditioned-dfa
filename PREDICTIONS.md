# Pre-registered predictions for in-flight experiments

Recorded before job completion (submitted 2026-07-09: jobs 29774322,
29774345, 29774356, 29774380, 29774381). The anisotropy law of the paper
(Proposition 1 + regime prediction, `drafts/Info-DFA/paper_main.tex`
§2.1/App. A) makes the following falsifiable predictions for these runs.
Interpretation commitments — including what a *failed* prediction would
mean — are stated alongside each.

## P1. Noisy ImageNet-100 deconfound (`imagenet100_noisy_deconfound_v1`)

With 40% symmetric train-label noise, label-noise error energy loads
high-variance activity directions, so the law predicts the conditioning
gain over raw block-DFA reappears **at scale**:

- **P1a (sign, primary):** at layer4 substitution, both conditioned
  variants (diagonal channel-nDFA and full-cov ZCA) beat raw block-DFA in
  val top-1, reversing the clean-protocol ordering (clean: DFA 77.8 ≥
  diag 77.3 ≈ ZCA 77.4). Predicted magnitude: a few pp, not MLP-sized —
  ResNet-18's BatchNorm already standardizes per-channel scale, so only
  the off-diagonal/cross-channel share of the mechanism remains.
- **P1b (depth trend persists):** the substitution-depth degradation does
  NOT disappear under noise — all-block remains far below layer4 for all
  local variants. The depth cliff is a finite-sample/credit-assignment
  failure, not a regime effect, so noise should move the
  conditioned-vs-raw delta, not the depth trend.
- **P1c (BP gap narrows):** the BP-vs-conditioned gap at matched depth
  shrinks relative to the clean protocol (noise hurts the exact gradient
  more than a regularizing local rule; cf. noisy-MLP tier).
- **If P1a fails** (conditioning still ≤ raw DFA at scale under noise),
  the scale×regime deconfound resolves AGAINST the regime explanation:
  the boundary is architectural/scale-driven, and the paper's claim must
  be narrowed accordingly. We commit to reporting this outcome.

## P2. DFA+BatchNorm baseline (`infodfa_bn_baseline_v1`)

BN standardizes per-unit activation variance — the forward-pass analogue
of *diagonal* conditioning. The paper's diagonal controls (Adam/diag-K:
diagonal square-root K +4.8 pp vs full K-nDFA +37.1 pp, nuisance-dominant)
predict:

- **P2a:** DFA+BN improves over raw DFA in every stressed regime (it
  captures the scale/diagonal share).
- **P2b (key):** nDFA (no BN) still beats DFA+BN by a large margin in the
  nuisance-dominant regime — predicted ≥10 pp — because the
  nuisance-dominant gain lives in off-diagonal covariance structure BN
  cannot see.
- **P2c:** on noisy Fashion-MNIST the DFA+BN vs nDFA gap is small (the
  paper already shows norm matching suffices there: scale is the binding
  constraint, not anisotropy).
- **P2d:** on the clean task-aligned control, DFA+BN ≈ DFA (no
  anisotropy to remove).
- **If P2b fails** (BN recovers most of the nuisance-dominant gain), the
  practical value of update-side conditioning collapses to "BN for
  networks that can't use BN", and the paper must say so.

## P3. Amortized covariance refresh (`infodfa_amortized_refresh_v1`)

Activity second moments drift slowly under stationary inputs, so:
- **P3a:** k=10 refresh matches k=1 (per-batch) final accuracy within
  ~1 pp on both the nuisance cell and noisy Fashion-MNIST.
- **P3b:** k=50 degrades gracefully (≤ a few pp), with the deficit
  concentrated in early-epoch dynamics when statistics change fastest.

## P4. Overhead benchmark (`benchmark_overhead_v1`)

- **P4a:** non-amortized nDFA per-step overhead on the paper's MLP
  configs falls in the 3–20× band vs BP (consistent with the
  development-run 4.7–18.8× figures); K-nDFA is higher than nDFA.
- **P4b:** k=10 refresh brings nDFA under ~2× BP per step.

These predictions were derived from the linearized analysis and the
existing controls only; no result files from the five runs above were
read before this commit.

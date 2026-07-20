# Pre-registered predictions for in-flight experiments

> **Terminology note (added after the freeze):** References below to an
> “anisotropy law” preserve the preregistered wording. The current manuscript
> treats this as a signed empirical hypothesis, not a general law.

Recorded on 2026-07-09, after submission and before completion of the
corresponding runs. The anisotropy law of the paper (Proposition 1 + regime
prediction, §2.1/App. A) makes the following falsifiable predictions for
these runs.
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

## P5. MLP-Mixer with token-level conditioning (planned: `infodfa_mixer_v1`)

Registered before the Mixer runner exists. A small MLP-Mixer on noisy-label
CIFAR-10, with DFA feedback to token-mixing and channel-mixing layers and
nDFA conditioning by the corresponding presynaptic second moments
(token-level factor = the direct analogue of the channel factor):

- **P5a:** conditioned DFA beats raw DFA on noisy-label CIFAR-10 in the
  Mixer, i.e. the mechanism survives the token/channel factorization of
  mixer-family architectures.
- **P5b:** with LayerNorm enabled (standard Mixer), the relative gain is
  SMALLER than in the un-normalized MLP tier — LayerNorm partially
  standardizes per-token statistics — but remains positive, because
  cross-feature covariance is untouched by LayerNorm.
- **P5c:** with LayerNorm ablated, the gain grows toward MLP-tier size.
- **If P5b fails with zero/negative gain**, the law implies LayerNorm
  already provides the usable share of conditioning in normalized
  architectures, and the method's practical scope narrows to
  non-normalized (e.g. neuromorphic) settings; we commit to reporting it.

## P6. Clean Fashion-MNIST error-factor replication (`dfa_stall_fashion_threefactor_*_v1`)

Registered 2026-07-14 before any development or confirmation endpoint was
read. This is a dataset replication of the clean MNIST DFA-stall experiment,
not an input-noise manipulation. The architecture, optimizer, one-vs-rest
binary log loss, 1,000-step budget, layerwise gradient-norm matching, and
train/validation/test discipline are unchanged. A fixed 5,000-example split
(split seed 24680) selects the activity and error dampings independently from
`{0.03, 0.1, 0.3, 1, 3, 10, 30, 100}` on model/data-order seeds 60--62 and one
feedback seed. K-nDFA combines those two values without joint tuning.
Confirmation freezes them on model/data-order seeds 70--74 crossed with three
feedback seeds; feedback seeds are averaged within model seed before paired
comparisons, and the Fashion-MNIST test set is evaluated only after the final
update.

- **P6a (primary):** error nDFA improves mean clean test accuracy over raw DFA,
  with the paired difference positive for at least four of five model seeds.
- **P6b (complementarity):** K-nDFA improves mean test accuracy over activity
  nDFA. Test cross-entropy is a prespecified secondary endpoint and must be
  reported whether or not accuracy improves.
- **P6c (source swap):** K-nDFA using its local DFA-error covariance is within
  0.5 percentage points of the otherwise matched nonlocal BP-error-source
  control in mean accuracy, with no deficit of the same sign in all five model
  seeds. This tests the covariance source, not whether K-nDFA beats BP.
- **Failure interpretation:** if P6a fails, the paper will retain the MNIST
  result as a single-setting proof of concept rather than claim dataset-level
  error-side replication. If P6b fails, the two-sided extension will be framed
  as a formal symmetry whose incremental empirical value remains unresolved.
  If P6c fails, the source-swap control will be reported as evidence that local
  DFA errors are an inferior covariance source in this setting.

## P7. ReLU/softmax architectural replication (`dfa_relu_mnist_threefactor_*_v1`)

Registered 2026-07-15 after validation-only protocol development and before any
test-set evaluation for this experiment. This changes both architecture and
loss relative to P6: a two-hidden-layer 256/128 ReLU MLP uses multiclass softmax
cross-entropy for 1,000 SGD updates on clean MNIST. Every conditioned hidden
weight gradient is layerwise norm-matched to raw DFA. A fixed 5,000-example
training-validation split (split seed 86420), model/data-order seeds 0--2, and
one feedback seed selected activity damping from
`{0.03, 0.1, 0.3, 1, 3, 10, 30}` and error damping from
`{0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}`. The frozen values are
`lambda_A=3` and `lambda_E=0.1`; K-nDFA combines them without joint tuning.
Confirmation uses fresh model/data-order seeds 100--107 crossed with feedback
seeds 0--2. Feedback seeds are averaged within model seed before paired tests,
and the test set is evaluated only after the final update.

The dataset was chosen after an explicitly exploratory ReLU/softmax Fashion-
MNIST development pilot: independent factor selection there did not preserve a
K-nDFA increment. P7 is therefore a prospective fresh-seed confirmation of the
MNIST architectural replication, not a preregistered dataset-generalization
claim. The Fashion pilot will not be presented as confirmatory evidence.

- **P7a (error side):** error nDFA improves mean test accuracy over raw DFA,
  with a positive paired difference for at least six of eight model seeds.
  Test cross-entropy is reported as a secondary endpoint; development indicates
  that accuracy can improve even when error-only calibration worsens.
- **P7b (two-sided complementarity):** K-nDFA improves mean test accuracy over
  activity nDFA, with a positive difference for at least six of eight model
  seeds, and lowers mean test cross-entropy.
- **P7c (architecture/loss scope):** P7a and P7b together count as replication
  of the two factor-specific signs under ReLU/softmax, not as evidence of
  superiority to BP or of dataset-wide generality.
- **Failure interpretation:** if P7a fails, error-only improvement remains
  specific to the tanh/one-vs-rest confirmations. If P7b fails, K-nDFA's
  incremental evidence remains confined to that architecture/loss. Either
  failure will be reported, and the title/abstract will be narrowed rather than
  using the development endpoints as evidence.

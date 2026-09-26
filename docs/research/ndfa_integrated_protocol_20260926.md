# Integrated development and one joint confirmation

Frozen before training the new development studies. The user authorized the merged plan on 26 September UTC / 25 September Eastern. This replaces the unexecuted test-evaluation stages of `decision_round_v4`; it does not overwrite its configurations or outcomes.

## Existing evidence and the amendment

At the amendment, the fixed-epoch and constant-rate cohorts had completed all 80 and 128 training runs; the geometry development had completed 40, and matched-work development had completed 114 of 192. The geometry selector had already launched its 70-run cohort. All remain in the record under their original roles and source hashes. **No official-test evaluation plan existed for any of these studies.** Three pending test-evaluation jobs and the pending work selector were canceled. The amendment receipt is in the prior round's `integrated_revision_amendment.json`.

The expanded investigation may use these cohorts' validation evidence as documented development/intervention evidence. They will not be represented as confirmations of the subsequently expanded hypotheses. The joint confirmation uses new seeds 940101–940110. Previously inspected official CIFAR/MNIST/Fashion test data are historically exposed; new seeds do not make those datasets newly untouched. The original MNIST-background-images test member remains unread until the global evaluation gate.

## Parallel development allocation

| Study | Runs | Learning-work budget per run | Purpose |
|---|---:|---:|---|
| FOOF on CIFAR-10 | 102 | 200 H100 seconds | BP/DFA × SGD/FOOF, plus tuned AdamW, A, FD and unnormalized DFA references |
| FOOF on the synthetic nuisance cell | 156 | 20 H200 seconds | Same credit/optimizer factorial with and without BN, plus tuned references |
| Temporary error factor, CIFAR-10 | 60 | 200 H100 seconds | A, early K→A, persistent K, early diagonal-E→A |
| Temporary error factor, MNIST ReLU | 60 | 60 H200 seconds | Prespecified secondary replication; cannot replace a failed CIFAR primary |
| Seven benchmark conditions | 504 | 60 H200 seconds | Two foreground datasets × clean/label noise/input nuisance, plus original MNIST-background-images |

Total: **882 new development runs**, two paired development seeds per candidate. Seven short GPU checks precede substantive runs. Training arrays have throttle 60 on requeue. H100 priority and eng can be used with the same accelerator type when available; scheduler migrations do not change a scientific setting. No other project is canceled or modified. No grid is enlarged after an edge selection.

Selection uses mean minimum validation CE across the two declared seeds, followed by validation accuracy and case ID. Every eligible run must terminate with finite results; numerical failures remain in the inventory. Every candidate receives the same validation opportunities at approximately 5% work increments, including initialization and the final single-step overshoot. FOOF calibration, conditioning, inverse updates, learning and numerical checks are charged. Validation, fixed-state probes, hashing and checkpoint exports are separately excluded.

FOOF uses normalized 0.95 EMA moments, 50 calibration batches, inverse refresh every 100 updates, all weight layers including the readout, no norm matching, and unconditioned bias/BN-affine SGD updates. It adapts the published recipe to our biased/BN architecture; it is not an exact reproduction of Benzing's experiments. The factorial compares raw SGD and FOOF within BP/DFA; additional AdamW baselines prevent a weak-SGD performance headline. The two credit rules receive the same factorial grids. A conditions hidden weights only and preserves its existing layerwise norm convention.

## Benchmark predictions and provenance

The constructed MNIST/Fashion tasks use identical 10,000 training and 2,000 validation foreground identities across their three conditions, sampled from `train=True` only with seed 934005. Clean pixels use division by 255. Label noise independently changes 20% of training labels to a different class; validation and test labels remain clean. Input nuisance is a fixed 0.35 foreground + 0.65 independent grayscale CIFAR-training-image background mixture. Training, validation and prospective test background pools are disjoint, with identities and source hashes recorded. Background assignment does not use the foreground class. These are **constructed paired variants**, not the original Larochelle benchmark.

The independent standard task uses the original publisher archive `https://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip` (SHA256 `1b6318b8774bf1a96480c32afb68147b6b9d529c3b646a0a739d9f0e11b580c3`). Its 12,000-row training member is divided into the first 10,000 training and last 2,000 validation examples. The 50,000-row test member is not read in preparation or development. This benchmark has independent foreground provenance and is not treated as a paired image intervention with the constructed clean condition.

Before outcomes, predict that activity DFA gains accuracy and lowers CE relative to tuned DFA under input nuisance, and that its accuracy advantage is larger there than on clean data. Predict a small accuracy difference on clean data and label noise alone, with a prespecified ±1 percentage-point equivalence margin. The standard background task predicts a positive activity-DFA benefit in accuracy and CE. These are falsifiable working hypotheses motivated by the model, **not consequences guaranteed by a task's label**. Report all conditions, including reversals and failures of equivalence. A nonsignificant difference does not establish a small effect.

Each condition includes DFA+BN, BP+BN, A+BN, centered-A+BN, FOOF-BP+BN, FOOF-DFA+BN, FD-DFA and unnormalized DFA during development. Learning-rate/damping grids are explicit in `designs.json`. Confirmation retains the six prespecified practical families (all except centered A and unnormalized DFA); the excluded families remain development-only regardless of whether they win. Centered covariance receives its separate full geometry confirmation. CIFAR-10 already supplies the non-saturating real task; another CIFAR-100 expansion is deferred beyond this finite decision round.

## Error-factor test and stopping rule

All error studies use **constant learning rate**, with no warmup or decay. A uses fixed activity damping (relative 1 on CIFAR; absolute 0.3 on MNIST). Choose the common learning rate from A alone among 0.0001, 0.0003 and 0.001. Full and diagonal error operators each have relative damping 1, 10 and 100. Error moments use per-example gated local errors, undoing mean-loss normalization exactly once. Early E occupies the first quarter of measured learning work, after which activity conditioning continues. Persistent K uses the early-K-selected error damping. Biases, readout, BN-affine parameters and optimizer conventions are common to the A/K comparisons.

At common states and fixed training batches, measure the cosine between K and **A**, not between K and raw DFA: the latter would confound the two factors. Use the mean directional change `1 − cosine(K, A)` across hidden layers and the prespecified positive-time probes before the quarter-budget boundary, then aggregate within seed. Undefined directions make this criterion unavailable. A cosine of 0.999 or greater does not meet the threshold. This is an operational materiality threshold, not a universal theoretical boundary.

A positive scalar left factor cancels under the shared hidden-matrix norm matching, so its comparator is algebraically A. The implementation uses that identity, verified numerically; it is not an additional independent training replication or statistical test. Diagonal E remains a separately tuned directional control.

**E stays eligible for a main-text conditional claim only if the joint fresh-seed CIFAR confirmation passes all five tests:** early K improves accuracy and CE over A/scalar; it improves both over separately selected diagonal E; and its mean directional change exceeds 0.001. All five enter the global multiplicity family. MNIST is secondary and cannot substitute for a failed CIFAR primary. A nearly scalar selected operator fails the direction criterion. If the scientific gate fails, demote E to a short appendix note; no additional scientific rescue round, noise family, window or seed extension is allowed. Technical interruption is recorded separately and cannot be replaced with selectively successful runs.

## Shared-checkpoint interventions

The completed constant-rate study provides a validation-only gate: run branches if DFA early conditioning at the declared 0.0003 rate has higher mean final accuracy and lower mean final CE than late conditioning. No significance-based selection or rate switching is used. If the gate fails, report it and skip the branches.

If it passes, run 32 prefixes: BP/DFA × always/never activity conditioning × four new development seeds, on CIFAR and the synthetic nuisance cell. Prefixes use BN, AdamW 0.0003, activity relative damping 1 and 200 epochs. Save complete model, optimizer, normalization, RNG, minibatch cursor, history and cost state at epochs 50 and 150. At each checkpoint fork conditioning on and off for the next **50 epochs**, giving 128 continuations. Both branches share their parent state and future samples exactly. They differ only in whether activity conditioning is applied. The continuation clocks have equal constant learning rates and update counts.

For the synthetic task, probe 64 fixed task angles, each with eight independently redrawn nuisance vectors, at a fixed evaluation state. Record hidden variation under nuisance changes, variation across task angles, and their ratio alongside validation learning. This avoids projecting hidden coordinates into an unrelated input-space subspace. The probe never changes learning state. Reduced hidden variance alone is not evidence of preserved task information, and a branch outcome does not automatically establish irreversible nuisance capture.

## One joint confirmation and global test gate

After all new development, prior matched-work development and any required branches finish, write one immutable joint manifest binding every selected configuration, source hash, seed and endpoint. No individual test evaluation runs before every included confirmation cohort has a terminal audited record. No test result selects settings, cohorts, horizons or claims. A numerical failure remains visible; primary contrasts require all ten declared pairs. Work-budget primary contrasts additionally require uninterrupted runs.

The maximum joint inventory is 880 training runs: 100 CIFAR practical/factorial; 140 synthetic factorial; 40 CIFAR E; 40 MNIST E; 420 benchmark runs; 70 geometry; 40 constant-rate timing; and 30 fixed-epoch temporary-activity runs. An unavailable development family is explicitly missing; it cannot be replaced by a different task. The main practical endpoint is the checkpoint with lowest validation CE. Final checkpoints are secondary, except that the fixed-epoch geometry, timing and temporary-activity noninferiority endpoints are final by construction.

The global primary family comprises 21 two-sided paired tests: three CIFAR practical gains (A, early A and FOOF-DFA against tuned AdamW DFA), one BP/DFA FOOF-gain interaction, two geometry controls, two timing tests, five E tests, three input-nuisance tests per foreground dataset, and two standard-background tests. Use Holm correction across the entire family, including p=1 for unavailable comparisons. Accuracy differences are percentage points; positive loss contrasts mean lower CE. Individual 95% Student-t intervals and every paired value are retained. The ±1-point clean/label-noise equivalence checks are explicitly secondary individual checks, not an uncorrected global claim of equivalence.

A practical CIFAR claim additionally requires a mean gain of at least one percentage point and no increase in mean test CE. Temporary activity efficiency uses a separate one-sided 95% noninferiority bound greater than −0.5 points and mean paired work ratio at most 0.8. These are distinct from the E gate. No optional seed extension follows a borderline result.

Report synchronized learning time and **estimated leading-order linear-algebra FLOPs**, with FMA=2 and explicit costs for matrix products, Gram systems, dense solves, FOOF calibration/inverse refresh and FD transforms. The estimate excludes pointwise activation/BN/optimizer arithmetic, data processing, probes and exports; it is not measured hardware FLOPs or an assertion of FLOP matching. Equal-time comparisons retain the same GPU model within each study. Both work and arithmetic are reported so solver overhead cannot be hidden.

The automatic report is `results/ndfa_strengthening_20260925/integrated_round_v1/decision.md`, with seed-level metrics and retained predictions. Title, mechanism language and venue are chosen after scientific review of the complete evidence. No gate guarantees conference acceptance.

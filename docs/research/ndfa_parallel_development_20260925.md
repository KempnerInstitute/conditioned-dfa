# Parallel baseline development — September 25, 2026

The user authorized `kempner_eng` and `kempner_requeue` in addition to
`kempner_h100_priority`, and requested independent rounds in parallel. This
extension accelerates E1 of the strengthening plan. Mean/covariance controls,
alignment interventions and larger applications still depend on its outcome.

## Submitted studies

| Study | Candidates × paired development seeds | Training horizon | Hardware / partition | Slurm jobs |
|---|---:|---:|---|---|
| Existing CIFAR-10 stable baselines | 108 × 2 | 200 epochs | H100 / priority and requeue | Original verification 48550331; main 48554715; migration 48564083 |
| Tuned synthetic controls | 384 × 2 | 500 epochs | H200 / eng | Verification 48567324; remaining 48569415 |
| CIFAR-10 EMA FOOF-BP | 24 × 2 | 200 epochs | H100 / requeue | Verification 48567356; remaining 48569420 |

These are validation-development cohorts. The synthetic runner generates no
test split, and the CIFAR runner loads only the official training set. Two
development seeds are for hyperparameter selection; they are insufficient for
the paper's final uncertainty estimates. Confirmation settings, seeds and work
budgets will be frozen after inspecting boundary optima and learning curves.
Neither fixed horizon establishes convergence by itself.

Each new full array depends on successful execution of its GPU verification
array. Before training, every task checks the verification seeds' complete
update counts, finite final validation losses, and source/configuration/model
hashes. The gate has no accuracy or treatment-benefit threshold. A numerical
failure in a verification case requires inspection; it is not silently replaced
by a favorable result. Subsequent grid failures remain part of the study record.

## Scheduling and recovery

The original priority array's concurrency increased from four to eight.
Thirty-two **unstarted** cases (75–86 and 88–107) moved to a separate H100 requeue
array with concurrency four. Their original pending tasks were cancelled after
holding and checking them; running tasks and scientific configurations were
preserved. A per-case lock prevents concurrent writes. Its restart wrapper
verifies and skips completed seeds, archives interrupted attempts, and restarts
an interrupted seed from its original initialization. This older runner does
not resume partway through a seed.

The new synthetic and FOOF runner checkpoints every five epochs. Checkpoints
include model and optimizer states, FOOF moments and cached inverses, sampling
and augmentation RNG states, learning-rate position, histories and stream
digests. An interruption resumes at the last complete checkpoint. Atomic writes
protect checkpoints; each attempt records its device and Slurm partition.
Completed seeds are checked and skipped. Repeated infrastructure interruptions
eventually stop for inspection rather than retry indefinitely.

The synthetic array has concurrency four; the FOOF CIFAR array has concurrency
two. Slurm prohibits including `kempner_requeue` in a multipartition submission,
so these are separate arrays. Synthetic development stays on H200, and CIFAR
development stays on H100. Hardware timings are not pooled across these studies.
Primary timing claims will use a separate frozen, matched-hardware confirmation.

## Synthetic comparison

The generator reuses the original latent-circle data formulas with four task
coordinates, 24 nuisance coordinates, a random projection to 64 inputs and eight
classes. Four representative cells cover nuisance-dominant, low-sample, mixed
and task-aligned conditions. Their exact scales and corruption rates are in the
configuration. Training sizes are 512, 512, 1,024 and 4,096 respectively; each
validation set has 4,096 examples with clean labels.

Seeds 926101 and 926102 each determine paired data, initialization and feedback.
Independent `SeedSequence([seed, stream])` streams construct the projection,
training examples, training-label corruption and validation examples. A fifth
stream is reserved for future test generation and is unused. Changing validation
set size therefore cannot change the training samples or corruptions.

Every cell compares eight families: BP, DFA, activity-conditioned DFA, DFA+BN,
activity-conditioned DFA+BN, BP+BN, norm-matched BP+activity, and EMA FOOF-BP.
Each receives twelve configurations and the same two seeds. The first seven
families use SGD with momentum or AdamW and method-specific learning-rate
ranges. Activity families cross three rates per optimizer with absolute damping
0.03 or 0.3. FOOF crosses four SGD rates (0.003, 0.03, 0.3, 3) with three damping
values (0.03, 0.3, 3). Equal candidate counts do not imply identical optimizer
search spaces; all grids and failed candidates will be reported.

Models have hidden widths 256 and 128, batch size 128, a five-epoch warmup and
cosine decay to one percent of the peak learning rate. Activity conditioning
uses current-batch uncentered moments, hidden weights only, and norm matching to
each layer's own raw update. Bias and BN-affine gradients pass through. Synthetic
activity damping is absolute; the earlier CIFAR activity grid uses damping
relative to mean activity energy. These are distinct documented protocols.

The predefined selection rule is lowest mean **final validation cross-entropy**
over both seeds, with mean accuracy and then case ID as tie breakers. Incomplete
or numerically failed candidates are ineligible but retained. Later analysis
must inspect late training/validation curves and edge-of-grid winners before
freezing confirmation; a final-horizon development winner is not automatically
an early-stopping or near-convergence result.

## EMA FOOF comparator

The activity-side inverse is established prior art. This comparator follows the
EMA and inverse-refresh ordering of [Benzing, ICML 2022, Algorithm 1, Appendix K](https://proceedings.mlr.press/v162/benzing22a/benzing22a.pdf).
It uses normalized EMA decay 0.95, 50 calibration minibatches, moments updated
on every minibatch, and inverse refresh every 100 updates. The old inverse acts
on the current gradient; a scheduled refresh uses the previous moment estimate;
the current presynaptic activity then updates the EMA. In row-batch notation,
the moment is `A.T @ A / batch_size`; damping uses this mean-moment convention.

Every weight matrix, including the classifier, receives inverse conditioning
without norm matching. The experiment retains the shared architecture's biases
and optional BN-affine parameters and updates those with raw SGD gradients.
Benzing's original MLPs omitted biases; this is a documented architecture
adaptation, not an exact reproduction of those original experiments. Failed
factorizations at declared damping are numerical failures; damping is not
silently increased.

CIFAR compares FOOF-BP with and without BN, twelve configurations each, using
the same two seeds (925101/925102), 45,000/5,000 train/validation split (925005),
initialization, feedback and training augmentation/order streams as the existing
baseline grid. Hidden widths are 1,024 and 512, batch size 256, weight decay
0.0001 on weight matrices, and the horizon is 200 epochs. Calibration uses
separate streams and restores BN running statistics before training. It does
not consume the paired training-order or augmentation streams.

Measured training work includes calibration, sampling, augmentation, gradients,
moment updates, inverse refreshes, optimizer updates and finite-value checks.
Validation, checkpoint export and hashes are separately excluded. These are
development timings; interleaved jobs and preemption make a dedicated timing
confirmation necessary. Repeated work lost between checkpoints must not be
interpreted as saved algorithmic compute.

## Verification and artifact locations

Four operator tests compare FOOF against an independent dense recurrence,
including stale-inverse restoration and batch-mean normalization. Six paired
training tests verify exact uninterrupted-versus-resumed CPU model states and
validation results for raw, activity and FOOF updates, with and without BN.
Two additional tests check training/validation RNG separation and rejection of
a changed configuration. Two selection tests reject incomplete grids and check
CE-based selection, failed-candidate exclusion and boundary flags. All fourteen
passed in the public release worktree. Both synthetic GPU verification
cases completed both seeds at the full declared horizon and passed their
provenance and completeness gate.

The following directories contain immutable scientific source/configuration
snapshots, run manifests, validation histories and model checkpoints:

- `results/ndfa_strengthening_20260925/baseline_development_v1/`
- `results/ndfa_strengthening_20260925/synthetic_development_v1/`
- `results/ndfa_strengthening_20260925/foof_cifar_development_v1/`

Each new study has `submission.json` with exact case assignments and jobs, and
`remaining_dispatch_v1/` with its frozen launch gate. The baseline migration is
recorded in `requeue_dispatch_v1/migration.json`. The original synthetic snapshot
contains the initial multipartition Slurm default; the recorded submission
overrides it with `--partition=kempner_eng`. That rejected initial submission
started no training, and the scientific snapshot was preserved.

CPU summary jobs 48572523 (baseline), 48572524 (synthetic) and 48572537 (FOOF)
depend on completion of their training arrays. Their immutable analysis and
receipts are in each study's `summary_dispatch_v1/`. The summaries verify model
hashes and full update counts, retain all failures and per-seed values, and
select only after every candidate in a family has a terminal record. They flag
boundary choices and retain late learning curves for the next decision; they
do not mark a study ready for confirmation automatically. An initial partial
audit passed for the baseline and synthetic grids.

Configuration SHA-256:

```text
baseline:  0fbb4d7c4eedc6f067999814e505e10a9426e69a79b072d849aab4b933ce9489
synthetic: 2183261f9b2196d444a457654b5f75b82f6bc435543acf892be44d5f50898d70
FOOF:      b719ff2a3317b879ce083411db5df58929b14cfdbd197e8bcc9e116c69bff0b7
```

The next scientific decision follows complete development summaries: compare
stable loss scales and accuracy, inspect remaining hyperparameter boundaries,
then freeze confirmation and the targeted mean/covariance and alignment studies.
No new result is yet a manuscript headline or evidence of improved acceptance
odds.

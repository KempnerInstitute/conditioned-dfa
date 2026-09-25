# Second development round — September 25, 2026

The user requested the next jobs after the `kempner_eng` synthetic grid ended.
Its 768 runs produced 755 complete endpoints and 13 recorded numerical
failures. The full source, manifest, checkpoint and expected-update audit passed.
The resulting `round2_gate_summary.json` is the immutable input to this round.
All results below remain development results from two paired seeds. The
official test split is unused; no confirmatory claims or p-values follow from
this selection stage.

The first grid selects an activity benefit primarily in the nuisance condition.
The task-aligned condition has similar or weaker activity-conditioned endpoints
under these recipes. These descriptive findings motivate keeping both conditions
in the mechanism study. Seventeen of 32 cell/family selections touch a search
boundary, so final replication settings are not frozen yet.

## Submitted work

| Study | Configurations × seeds | Training runs | Partition, device, throttle | Training jobs | Summary job |
|---|---:|---:|---|---|---|
| Boundary extensions | 170 × 2 | 340 | `kempner_eng`, H200, 12 | 48592722 | 48594724 |
| Longer horizon | 32 × 2 | 64 | `kempner_requeue`, H200, 60 | 48592816 | 48594737 |
| Mean/covariance controls | 68 × 2 | 136 | `kempner_requeue`, H200, 60 | Verification 48593758; remaining 48593949 | 48594751 |

The total is **540 new training runs**, all using the original two synthetic
development seeds. Hardware stays H200 across these studies. Actual concurrency
depends on the scheduler. The geometry array waits for complete, finite GPU
verification endpoints for each of its five operators, with source/configuration
and checkpoint hashes verified. All ten verification runs passed this gate.
The inherited runner checkpoints every five epochs and resumes after preemption.

## Boundary extensions

Each affected cell/family retains its selected optimizer. For each selected
learning-rate or damping boundary, the search extends outward by factors of
three and ten. The new coordinates are crossed with the selected and immediately
adjacent old coordinates. Already-run points are excluded. The same trigger and
construction apply to BP, DFA and FOOF; the largest family extension adds twelve
configurations. The 17 affected groups add 170 configurations in total.

The 500-epoch horizon, data, initialization, order, schedule and scientific
runner are unchanged. The new configuration snapshots copy the original source
without editing it. Existing results remain immutable. Case IDs continue after
the original grid so pooled analyses cannot confuse two different settings.

Selection pools the original and new candidates. It uses lowest mean final
validation cross-entropy across both development seeds, then accuracy and case
ID as tie breakers. A candidate with a failed seed is ineligible but remains
reported. Every declared candidate in a family must finish before its pooled
selection is reported. The pooled summary flags remaining boundaries rather
than declaring confirmation readiness automatically.

## Longer horizon

Every one of the 32 original cell/family winners is trained for 1,000 epochs
instead of 500. Each run restarts from the original paired initialization; its
cosine schedule is stretched to the longer horizon. This is **not** continuation
from the 500-epoch model, and it does not establish convergence by itself.

The analysis compares paired validation accuracy and loss, complete learning
curves and training work at the two horizons. It retains families with a weak or
absent activity benefit. These are provisional first-grid recipes; the boundary
extensions may subsequently select different settings. Horizon outcomes do not
silently change the separate fixed-horizon tuning criterion.

## Mean/covariance controls

This is the synthetic part of the proposed mechanism study: nuisance and
task-aligned data, each with and without pre-ReLU BN. The CIFAR component awaits
its stable-baseline review. Each condition uses the first grid's activity-DFA
optimizer and learning rate as a provisional common recipe. It is an exploratory
operator comparison, not the final best-recipe comparison or independent
confirmation.

Let `mu = mean(h)` and `C = mean((h-mu)(h-mu)^T)`. The five operators use:

| Name | Moment supplied to the damped inverse |
|---|---|
| Full | `mean(h h^T)` |
| Diagonal | `diag(mean(h h^T))` |
| Centered | `C` |
| Diagonal covariance plus mean | `diag(C) + mu mu^T` |
| Isotropic covariance plus mean | `trace(C)/d I + mu mu^T` |

Every operator keeps the same raw gradient numerator, forward network, absolute
damping convention, hidden-layer update scope and norm matching to its own raw
update. Bias and BN-affine gradients pass through unchanged. The controls do not
center the forward activations. The isotropic covariance control preserves the
full moment's trace and mean term while removing centered covariance anisotropy.

All five operators receive three damping values: one third of, equal to, and
three times the selected anchor damping. This gives
`5 operators × 4 conditions × 3 damping values × 2 seeds = 120 runs`.
Two additional references per condition add 16 runs: raw DFA under the same
optimizer/rate as the activity anchor, and the original full-moment batch-space
implementation at the anchor damping. The raw matched-recipe reference is
distinct from the independently tuned DFA result in the baseline grid.

All five mechanism variants use dense feature-space solves, allowing their
operator behavior to be compared with a common implementation. The batch-space
reference checks the connection to the existing implementation. These prototype
timings are not an optimized speed comparison. Solves fail explicitly at their
declared damping; there is no adaptive damping rescue.

At each evaluation, a fixed training-data minibatch from an independent RNG
measures mean-energy fraction, centered covariance participation rank, and
off-diagonal centered covariance energy. The probe uses training-mode BN and
restores the running statistics, forward cache and training state afterward.
It uses no labels and does not change either training RNG. These are minibatch
diagnostics, not population covariance estimates. Probe work is excluded with
validation from measured training work.

The analysis retains matched-damping contrasts and separately selected operator
summaries, with individual seed values. A benefit of the mean-preserving control
would support a mean-related explanation; a residual advantage of the full rule
would motivate a more targeted correlation test. Neither outcome is assumed.

## Verification and provenance

New tests check full-operator agreement with the existing sample-space rule,
the distinction between uncentered moments and centered covariance at nonzero
means, all five dense controls against independent matrix formulas, and exact
training equivalence with and without diagnostic probes. Interrupted geometry
training restores the same model and validation outcome as uninterrupted
training. Design tests check symmetric boundary expansion, exclusion of duplicate
points, inclusion of every family in the horizon study, and retention of old
winners in pooled selection. The operator/runner/design suite and the additional
selection tests passed before submission.

All configurations, scientific source snapshots, manifests and checkpoints are
under `results/ndfa_strengthening_20260925/` in:

- `synthetic_boundaries_v1/`
- `synthetic_horizon_v1/`
- `synthetic_geometry_v1/`

Each has its parent summary, submission receipt and frozen analysis in
`summary_dispatch_v1/`. Summary jobs run after their training arrays finish and
report missing runs explicitly. Configuration hashes are:

```text
boundaries: 32915881bbe9d47840585c60d32cd4b967349c93aa4bbc604084d94db806e926
horizon:    e0d1e2244dbf210bb23af3d0a06aba05e2fddc9d5308c53c9691dfadb784641d
geometry:   86f990cbe230004ea8705d19e9c3309acff565e2dad0565801f6bff134536ea5
```

Independent-seed confirmation, causal alignment interventions, factor-study
baseline development and the larger application remain subsequent work. The
current round informs those choices; it does not consume their confirmation
data or establish a new paper claim.

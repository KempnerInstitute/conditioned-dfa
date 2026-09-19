# Second review: scientific corrections and supplementary curation

This revision starts from manuscript `7fa1eddfe59415e1e6979e0cfda039c4f49d66d1`.
It preserves the main scientific structure and original measurements. No new
model training or test-set inference was performed. The public source tag for
this editorial and analysis revision is `ndfa-editorial-2026-09-19`.

## Corrections implemented

| Review concern | Implemented change |
|---|---|
| Full versus diagonal moments | Main text and Appendix E distinguish off-diagonal **uncentered** structure from centered covariance and mean-activity effects. Separate selection and unequal search remain explicit. No centered-correlation isolation is claimed. |
| Synthetic BN uncertainty | Each contrast is averaged over the 32 designed cells and retained feedback draws within each of five global seeds. SEM is computed across those five summaries, conditional on this design. |
| Hard CIFAR-100 crossed runs | Local-rule means average the five feedback draws within each initialization; SEM uses the resulting five initialization means, conditional on the feedback set. The final column is “Training runs.” |
| ColoredMNIST significance | Removed uncorrected significance symbols; retained the paired comparisons and the conclusion after Holm correction. |
| Strong-compression limit | States both rate-ratio limits and the sufficient condition that activity damping vanishes relative to task eigenvalues. |
| Flow clocks | Separates invariant minimum risk and within-rule timing ratios from absolute flow time; timing is not computational cost. Forward weights align to fixed feedback. |
| Figure 1D | Recovered and reproduced the population simulation, including initial weights, residual, moments, target orientation, damping, per-run rate search and stopping rule. Removed the unsupported condition-number-sweep reference. |
| Nuisance association | Figure and text use one 128-cell analysis, Spearman rho 0.623327. Its test-selected rank and exploratory scope are explicit. The prospective task-subspace basis has orthonormal columns. |
| Practical-study provenance | Added an eight-study ledger with development selection, final cohorts, hardware and artifact IDs. All three work budgets use H100 80GB; original 30-second TF32 flags were not recorded. Separate cohorts are not checkpoints of one trajectory. |
| Mechanistic overclaims | Routing failures do not exclude other spatial mechanisms; activation-routing and damping explanations are hypotheses. Clean CIFAR-10/CIFAR-100 sign differences are descriptive. Main text says unnormalized raw DFA is also unstable. |
| Figures and uncertainty | Follow-up panels show paired 95% t intervals and a CE tick at one. ImageNet has two spacious panels; the duplicated kernel-patch panel is removed. Captions distinguish three-seed SEMs, clean mean-only contrasts, and separate cohorts. |
| Bibliography and abstract | Added one practical quantitative anchor; alphabetized and standardized references. Dalm journal metadata agrees with Crossref. Removed the unverified Braun PMLR volume while retaining its ICML/arXiv identification. |

The BN means are unchanged. Corrected SEMs in percentage points are 0.69,
0.48, 0.30 and 0.28 for nuisance, low-sample, mixed and task-aligned regimes.
Hard CIFAR-100 accuracies are unchanged; crossed-run SEMs are replaced by the
conditional initialization-level summaries described above. These estimates do
not quantify uncertainty over new datasets or new hyperparameter searches.

## Supplementary structure and curation

The supplement now has 13 appendices: linear analysis; shared methods and
provenance; synthetic controls; independent factor confirmations; activity scale
transfer; error-factor transfer; matched work; targeted follow-ups; separate
predictive-stability protocol; constructed moment orientation; additional
architectures; ImageNet block substitution; and concise related mechanisms.
Each study keeps its methods, settings, results and limitations together.

Three secondary figures are removed from the active PDF: projected-step/rank
diagnostics, exploratory vision learning curves, and ImageNet routing
trajectories. Their necessary definitions and endpoints remain. The stale or
ambiguous error bands in those historical presentations are not current estimates.

Seven tables are removed: three exhaustive development grids, two secondary
all-pairs contrast tables, the duplicate cross-study endpoint summary, and the
constructed-mechanism endpoint table that repeated its figure. A cohort ledger
is added. Supplementary tables therefore decrease from 43 to 37. Redundant
contrast rows, repeated discussion and a broad literature survey are shortened.
Ten references used only in the removed survey are omitted. The complete paper
has 14 figures and 49 references. Supplementary source words decrease by about
17%; no font or margin changes were used to achieve this.

The removed tables and figures remain recoverable from the baseline commit;
experimental records remain in the research/reproduction archives. The curated
PDF retains the failed prospective predictor, protocol-dependent instability,
strong BP baseline, limited error-factor transfer, and other results necessary
to assess the claims. Curation removes duplication and development detail, not
inconvenient evidence.

## Verification and reproducibility

- `scripts/ndfa_revision_20260919/verify.py` independently checks the compact
  corrected summaries, common correlation, follow-up intervals and mathematical
  qualifications.
- `prepare_evidence.py` exports summaries from the original saved records;
  `linear_simulation.py` reproduces the small population illustration.
- `redraw.py` is the current figure entrypoint, using the previous generator for
  unchanged panels and the corrected evidence for changed panels.
- The two legacy table generators now implement the same seed-level definitions.
- The manuscript build checks official styles, all references, overflow, figure
  bounds, the nine-page main limit, author order and appendix ownership of floats.
- The arXiv package is independently compiled outside the workspace and compared
  against the reviewed PDF in text and rendered pixels.
- The anonymous evidence archive is extracted outside the workspace, verified,
  rebuilt and scanned; regenerated figures must match the active PDF assets.

See `ndfa_second_review_artifact_map_20260919.json` for final artifacts and
receipts. Earlier revision documents describe historical releases. No public
arXiv replacement or conference submission is performed by these build steps.

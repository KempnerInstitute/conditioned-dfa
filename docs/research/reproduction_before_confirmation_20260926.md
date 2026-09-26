# Reproducing the second-review revision

The [artifact map](docs/research/ndfa_expanded_figures_artifact_map_20260919.json) binds
this revision to exact source and evidence hashes. The manuscript and code
repositories share the tag `ndfa-expanded-figures-2026-09-19`. The main ICLR text,
including every main figure and table, fits within nine pages in the official
style. The full documents have 51 pages (ICLR) and 49 pages (author preprint),
with 14 active figures and 49 references.

## Build the paper

From this code repository, obtain the separate manuscript sources:

```bash
git clone --branch ndfa-expanded-figures-2026-09-19 https://github.com/houman1359/Info-DFA-draft.git drafts/Info-DFA
python -m pip install -r requirements.txt
python drafts/Info-DFA/scripts/build_manuscript.py
python drafts/Info-DFA/scripts/build_arxiv_package.py --output-dir drafts/Info-DFA/build/arxiv_upload --reference-pdf drafts/Info-DFA/build/checked/conditioned_dfa_arxiv.pdf
```

The build requires pdfLaTeX and the packages in `paper_preamble.tex`. The checker
verifies reviewed source/style hashes, both three-pass builds, references, figure
positions, anonymity, author order and the complete main-text page limit.
The arXiv builder compiles an independent extraction and compares its full text
and rendered pages with the reference PDF. It creates upload files without
submitting them. Use a new output directory for each package.

## Verify the evidence archive

Obtain the separately prepared `iclr_supplement.zip`, check its SHA256 against
the artifact map, extract it into a new directory and enter `review_package`:

```bash
python -B reproduce.py --checks integrity,statistics,legacy,focused,paper,smoke
python -B verify_revision.py
python -B scripts/ndfa_revision_20260919/verify.py
python -B -m pytest -q -p no:cacheprovider tests/test_ndfa_revision_math.py
```

The first command checks retained evidence and runs 143 CPU tests. The second
checks all 202 new cases (32 validation-only, 170 final-test models), whole-seed
intervals, the additional eight-seed diagonal contrast, loss tails and the
complete-risk counterexample. Three mathematical tests additionally check the
corrected calculation. The second-review verifier checks the remaining BN and
CIFAR-100 seed-level summaries, the common 128-cell nuisance correlation,
paired follow-up intervals, and the theoretical qualifications. The archive retains every planned case; large finite
losses are not silently dropped.

The original independent audit recomputed all new metrics directly from saved
logits in float64 and verified checkpoint hashes. To keep the review archive
below 100 MB, its new prediction exports contain per-example cross-entropy
rounded to float32, plus exact predicted and true classes. Source and export
hashes are distinct. Replaying these summaries is not a fresh inference audit.
Raw logits and model/optimizer states remain in the research workspace.

For the smaller mathematical/cohort checks in this repository:

```bash
python -m pytest -q tests/test_ndfa_revision_math.py tests/test_covariance_power_cohorts.py tests/test_replication_figure_cohorts.py
```

## Verify the second-review compact summaries

The new compact CSVs in `assets/ndfa_revision_20260919/` are available in this
source repository as well as the review archive:

```bash
python scripts/ndfa_revision_20260919/verify.py
python scripts/ndfa_revision_20260919/linear_simulation.py
```

The second command reruns only the small CPU population illustration for
Figure 1D and records its selected rates. `prepare_evidence.py` exports the
summaries from original raw archives; it requires those preserved inputs.
Seed-level SEMs are conditional on the designed cells and fixed feedback set.
See the study ledger for the distinct CIFAR-10 development/final cohorts.

## Regenerate figures and numerical analyses

Use the supplied evidence archive for the saved-data inputs; a source-only clone
cannot reconstruct measurements that are absent from Git. From its root:

```bash
NDFA_FIGURE_CACHE=revision/figure_cache.pkl NDFA_FIGURE_OUTPUT=/tmp/ndfa_figure_previews NDFA_PAPER_FIGURES=/tmp/ndfa_figure_pdfs python scripts/ndfa_revision_20260919/redraw.py
NDFA_FIGURE_CACHE=revision/figure_cache.pkl python scripts/ndfa_figure_layout_20260919/redraw.py --output /tmp/ndfa_compact_previews --paper-figures /tmp/ndfa_figure_pdfs
python analysis/validate_mode_timing.py
```

The first command produces 13 revised plots; the second applies the final
layouts to all five main figures and the supplementary normalization figure.
It checks retained measurements and all 21 Figure 5 paired contrasts against
the preceding generator and source summaries. The third command produces
the corrected mode-timing figure and numerical comparisons. The cache contains locally
constructed Matplotlib figures, with obsolete panels replaced from audited
measurements by the generator. Its hash is recorded; use only the supplied
artifact. Full-workspace audits are `audit_saved.py` and `audit_followups.py`
in `scripts/ndfa_revision_20260918/`. They require preserved original inputs,
including checkpoints, beyond the compact export.

## Reproduce the targeted GPU studies

The [frozen protocol](docs/research/ndfa_revision_followups_protocol_20260918.md)
specifies the complete inventory, validation-only selection, independent seeds,
work budgets and fixed-recipe interventions. Training source hashes and historical
paths are immutable provenance. Prepare a new relocated copy rather than editing
the original plan:

```bash
python scripts/ndfa_revision_20260918/prepare_rerun.py --data-dir /path/to/cifar10 --destination /path/to/new_ndfa_rerun
cd /path/to/new_ndfa_rerun
python scripts/ndfa_revision_20260918/runner.py --plan configs/reproduction_plan.json --phase work
python scripts/ndfa_revision_20260918/runner.py --plan configs/reproduction_plan.json --phase width
python scripts/ndfa_revision_20260918/runner.py --plan configs/reproduction_plan.json --phase stability
```

Run those commands on an allocated GPU, not a login node. The recorded jobs used
`kempner_h100_priority`; the [Slurm template](slurm/ndfa_revision_20260918.sbatch)
shows their resource settings. Data and software versions must be recorded for
any new execution. Wall-clock results depend on the hardware; the transferred
60/120-second settings were not retuned separately for each budget.

The earlier matched-work and moment-orientation studies have distinct protocols.
Their scripts are mapped in the README and retained review archive; do not pool
those cohorts with the new studies. Packaging scripts consume the preserved
base evidence archive and audits and are release-maintenance tools, not a
source-only download of all experimental data.

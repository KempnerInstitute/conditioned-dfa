# Conditioned Direct Feedback Alignment

Code for **Conditioned Direct Feedback Alignment via Activity and Error Geometry**,
by Houman Safaai, Varun Reddy, and Bernardo L. Sabatini.
The [manuscript repository](https://github.com/houman1359/Info-DFA-draft)
contains the anonymous ICLR entrypoint and author preprint sources.
The published [arXiv preprint](https://arxiv.org/abs/2607.18574) predates the
September 18–19 scientific revision described here.

Activity conditioning improves fixed-feedback learning in nuisance-dominated
settings and retains gains over DFA with shared normalization, wider MLPs and
matched measured work. Error conditioning provides a distinct benefit in the
clean factor confirmations but depends more strongly on damping and protocol.
BP remains stronger in the matched-work comparisons; forward decorrelation is
competitive. The paper studies update geometry under approximate credit, with
explicit limits on biological locality and practical generalization.

The September 25 correction release attributes the activity operator to FOOF,
qualifies the shared-rate synthetic gains using their loss distributions,
and discloses earlier test-probe exposure for the original MNIST cohort.
The supplement adds the direct within-BN comparison, exploratory Adam controls,
and explicitly separated post-hoc seed extensions. Synthetic uncertainty now
uses seed-level Student t intervals. The paper retains its compact figures.

Long-horizon validation development is underway under the
[baseline protocol](docs/research/ndfa_baseline_development_protocol_20260925.md).
These development runs are not new confirmed paper results. The full EMA-based
FOOF baseline, independent confirmation and mechanism studies remain pending.
Run the correction checks with:

```bash
python scripts/ndfa_strengthening_20260925/verify_corrections.py
python -m pytest -q tests/test_ndfa_strengthening_20260925.py
```

The current [figure revision](docs/research/ndfa_figures_3_5_revision_20260919.md)
compacts Figure 3 and expands Figure 5 to four panels, adding the existing
work and width follow-ups. Duplicate supplementary panels are removed.
The preceding horizontal layouts in Figures 1, 2 and 4 remain unchanged;
corrected measurements and uncertainty estimates are preserved.

The preceding [experimental revision](docs/research/ndfa_revision_findings_20260918.md)
remains integrated, including all 202 targeted follow-up cases. Original
experimental records remain available in the research and review archives.

## Reproduction and evidence

[REPRODUCE.md](REPRODUCE.md) gives build, verification, figure and training
commands. The [artifact map](docs/research/ndfa_expanded_figures_artifact_map_20260919.json)
binds source hashes, audited evidence, PDFs and upload archives to this revision.
The previous figure release is tagged `ndfa-expanded-figures-2026-09-19`. The correction and development work is on branch `ndfa-strengthening-20260925`; the paper uses branch `strengthening-20260925`.

| Claim or artifact | Implementation / record |
|---|---|
| Complete-risk calculation and seed-level resampling | [Mathematical functions](analysis/ndfa_revision_math.py), [mode-timing simulation](analysis/validate_mode_timing.py), [seed analysis](analysis/compute_infodfa_seedlevel_stats.py) |
| Longer matched work, width-specific damping and BN intervention | [Frozen protocol](docs/research/ndfa_revision_followups_protocol_20260918.md), [configuration](configs/ndfa_revision_followups_20260918.json), [runner](scripts/ndfa_revision_20260918/runner.py) |
| Saved-logit endpoints, loss tails and existing diagonal control | [New-study audit](scripts/ndfa_revision_20260918/audit_followups.py), [saved-evidence audit](scripts/ndfa_revision_20260918/audit_saved.py) |
| Earlier matched-work and forward-decorrelation comparison | [Protocol](docs/research/ndfa_bn_forward_decorrelation_protocol_20260915.md), [training and verification](scripts/ndfa_bn_forward_decorrelation_20260915) |
| Controlled moment orientation and credit quality | [Experiment](experiments/run_ndfa_factor_mechanism.py), [analysis](analysis/analyze_ndfa_factor_mechanism.py) |
| Seed-level BN / CIFAR-100 corrections and cohort provenance | [Checks](scripts/ndfa_revision_20260919/verify.py), [study ledger](assets/ndfa_revision_20260919/study_ledger.json) |
| Revised publication figures | [Compact overlays](scripts/ndfa_figure_layout_20260919/redraw.py), [base generator](scripts/ndfa_revision_20260919/redraw.py), [corrected numerical checks](scripts/ndfa_revision_20260919/verify.py) |

The compact anonymous evidence archive is prepared as a separate submission
attachment. Generated PDFs, the archive, datasets, raw logits and checkpoints
are not stored in this Git repository. Paths under `results/` in the artifact
map identify the preserved workspace/review-package evidence, not public GitHub
files. The compact archive verifies reported endpoints from saved per-example
losses and classes; it does not rerun GPU training, checkpoint inference or timing.

No arXiv replacement or conference submission was performed during this revision.
Earlier development documentation is historical; use the revision findings and
artifact map for current claims.

## Citation

```bibtex
@misc{safaai2026conditioned,
  title = {Conditioned Direct Feedback Alignment via Activity and Error Geometry},
  author = {Safaai, Houman and Reddy, Varun and Sabatini, Bernardo L.},
  year = {2026},
  eprint = {2607.18574},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2607.18574}
}
```

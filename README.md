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

The current second-review revision qualifies the full-versus-diagonal moment
interpretation, fixes the remaining seed-level uncertainty summaries, and
clarifies the theory's joint limit and flow clock. It organizes the supplement
into 13 appendices, removes duplicate/development presentations and supplies a
study ledger. The active paper has 14 figures and 49 references. No new model
training was needed. See the [second-review findings](docs/research/ndfa_second_review_revision_20260919.md).

The preceding [experimental revision](docs/research/ndfa_revision_findings_20260918.md)
remains integrated, including all 202 targeted follow-up cases. Original
experimental records remain available in the research and review archives.

## Reproduction and evidence

[REPRODUCE.md](REPRODUCE.md) gives build, verification, figure and training
commands. The [artifact map](docs/research/ndfa_abstract_artifact_map_20260919.json)
binds source hashes, audited evidence, PDFs and upload archives to this revision.
The current tag is `ndfa-abstract-2026-09-19` in both code and manuscript repositories.

| Claim or artifact | Implementation / record |
|---|---|
| Complete-risk calculation and seed-level resampling | [Mathematical functions](analysis/ndfa_revision_math.py), [mode-timing simulation](analysis/validate_mode_timing.py), [seed analysis](analysis/compute_infodfa_seedlevel_stats.py) |
| Longer matched work, width-specific damping and BN intervention | [Frozen protocol](docs/research/ndfa_revision_followups_protocol_20260918.md), [configuration](configs/ndfa_revision_followups_20260918.json), [runner](scripts/ndfa_revision_20260918/runner.py) |
| Saved-logit endpoints, loss tails and existing diagonal control | [New-study audit](scripts/ndfa_revision_20260918/audit_followups.py), [saved-evidence audit](scripts/ndfa_revision_20260918/audit_saved.py) |
| Earlier matched-work and forward-decorrelation comparison | [Protocol](docs/research/ndfa_bn_forward_decorrelation_protocol_20260915.md), [training and verification](scripts/ndfa_bn_forward_decorrelation_20260915) |
| Controlled moment orientation and credit quality | [Experiment](experiments/run_ndfa_factor_mechanism.py), [analysis](analysis/analyze_ndfa_factor_mechanism.py) |
| Seed-level BN / CIFAR-100 corrections and cohort provenance | [Checks](scripts/ndfa_revision_20260919/verify.py), [study ledger](assets/ndfa_revision_20260919/study_ledger.json) |
| Revised publication figures | [Generator](scripts/ndfa_revision_20260919/redraw.py), [corrected numerical checks](scripts/ndfa_revision_20260919/verify.py) |

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

# Conditioned Direct Feedback Alignment

Code and evidence for **Conditioned Direct Feedback Alignment via Activity and
Error Geometry**, by Houman Safaai, Varun Reddy, and Bernardo L. Sabatini.

The September 26 revision includes the completed joint confirmation: 880
training runs across eight cohorts, ten paired seeds, and 21 primary contrasts.
Activity conditioning and its established FOOF formulation improve tuned DFA
under matched measured training time. Covariance controls and early-only
conditioning support more specific conclusions about activity statistics and
training cost. The stable-training error-factor gate fails despite favorable
mean effects; the timing tests do not establish a DFA-specific mechanism, and
the image-background predictions are not confirmed. Equal measured time does
not imply equal FLOPs.

The [paper repository](https://github.com/houman1359/Info-DFA-draft) contains the
current manuscript, figures, and checked ICLR/arXiv build commands. The
[arXiv record](https://arxiv.org/abs/2607.18574) is updated separately; preparing
a source bundle does not replace that public record.

The focused arXiv correction release is tagged **`ndfa-arxiv-review-2026-09-26-v2`**
in both repositories. Its [revision guide](docs/ARXIV_REVIEW_20260926.md) maps
the frozen protocol, recorded timestamps and hashes, current figure inputs,
and publicly available evidence. It clarifies conditioning before AdamW and
corrects presentation without changing the results. The second correction
pass distinguishes the separate noninferiority criterion, narrows the Fisher
summary, and repairs notation and prose interruptions; figure data are unchanged.

## Verify the current results

```bash
python -m pip install -r requirements.txt
python scripts/ndfa_strengthening_20260925/audit_joint_decision.py \
  --root assets/ndfa_confirmation_20260926 \
  --output build/confirmation_verification.json
```

This verifies the exported file hashes, all 880 declared cases, all 21 primary
seed-level contrasts, individual t intervals, global Holm adjustment, and the
error-factor and efficiency gates. It uses supplied endpoint summaries and
does not retrain models or reevaluate checkpoints. See [REPRODUCE.md](REPRODUCE.md)
for CPU checks, source builds, and relocation of the frozen training protocol.

| Evidence or implementation | Location |
|---|---|
| Frozen settings, test summaries, source snapshot, and claim map | [Confirmation evidence](assets/ndfa_confirmation_20260926/README.md) |
| Activity and error conditioning | [Local preconditioning](infogeo/local_preconditioning.py), [integrated operators](scripts/ndfa_strengthening_20260925/integrated_components.py) |
| FOOF with either credit rule | [FOOF](infogeo/foof.py), [training loop](scripts/ndfa_strengthening_20260925/integrated_train.py) |
| Forward-decorrelation reference | [FD implementation](infogeo/decorrelated_dfa.py) |
| Validation selection, test gate, and declared comparisons | [Study control](scripts/ndfa_strengthening_20260925/integrated_control.py) |
| Independent endpoint reconstruction | [Audit](scripts/ndfa_strengthening_20260925/audit_joint_decision.py) |
| Earlier corrected theory and cohorts | [Mathematical checks](analysis/ndfa_revision_math.py), [September 25 evidence](assets/ndfa_strengthening_20260925) |

The public evidence contains every retained confirmation endpoint, including
direction diagnostics. Dataset files, model checkpoints, and per-example
prediction tensors are not included. Historical absolute paths and hashes in
the frozen records identify the original experiment; the portable audit and
rerun helper use local copies. Historical development notes describe their
own stages and must not replace the completed confirmation's conclusions.

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

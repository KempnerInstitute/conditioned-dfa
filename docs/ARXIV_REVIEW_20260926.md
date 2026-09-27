# Focused arXiv correction pass, September 26, 2026

Current release tag in both repositories: `ndfa-arxiv-review-2026-09-26-v2`.
This follows `ndfa-confirmation-2026-09-26`; it does not change the 880 training
runs, selected settings, 21 primary contrasts, claim gates, title, or abstract.
The manuscript repository is [Info-DFA-draft](https://github.com/houman1359/Info-DFA-draft).
Preparing this release does not update arXiv or the OpenReview submission.

## Second manuscript correction pass

The `-v2` tag incorporates the follow-up review of `Info_DFA_draft (9).pdf`.
Figure 5 now separates Holm-adjusted superiority tests from the independently
prespecified noninferiority/work criterion. The Fisher summary is restricted
to the stated regression model and distinguishes DFA pseudo-errors from a
Fisher factor. The manuscript also fixes the activity-moment alias, plain-SGD
scope, BN derivative in Algorithm 1, damping scope, and stale related-work
reference. Three local prose/float interruptions are repaired; the normalization
figure is placed with its intervention. Table 1 is larger in the preprint and
several compact tables have wider column spacing.

All figure PDF bytes, numerical data, training code, scientific settings and
primary statistical decisions are unchanged from `ndfa-arxiv-review-2026-09-26`.
That earlier tag and its source bundle remain intact. This code commit updates
only release/reproduction documentation to identify the current manuscript.

## Implementation clarified

A/E/K first transform the raw hidden-weight pseudo-gradient. Norm matching,
when used, is applied to that matrix. `integrated_train.py` then supplies it
to `step_optimizer` in `baseline_development.py`, which assigns `p.grad`
before calling AdamW. The optimizer's first and second moments and decoupled
weight decay subsequently determine the parameter step. Matching input matrix
norms does not match final AdamW step norms. `direction_probe` compares K and
A matrices at a common state **before** AdamW; it does not compare parameter
steps from different optimizer histories. No training implementation changed.

FOOF uses its own declared recipe: EMA moments, calibration, periodic inverse
refresh, all weight matrices including the readout, no norm matching, and SGD.
It is not the same AdamW recipe with a renamed credit signal.

## Exact frozen records

All links below are relative to this tagged repository. Original files are
unchanged, including historical absolute paths and recorded hardware amendments.

| Record | Location | Recorded time (UTC) / role |
|---|---|---|
| Development protocol | [protocol.md](../assets/ndfa_confirmation_20260926/protocol.md) | Before the expanded development; includes selection, claims and stopping rules |
| Joint confirmation specification | [joint_confirmation.json](../assets/ndfa_confirmation_20260926/joint_confirmation.json) | 2026-09-26 05:12:48.556997; selected cases, seeds and sources |
| Interpretation map | [claim_map.md](../assets/ndfa_confirmation_20260926/claims/20260926T074530Z/claim_map.md) | After development, before joint test evaluation |
| Interpretation-map receipt | [freeze_receipt.json](../assets/ndfa_confirmation_20260926/claims/20260926T074530Z/freeze_receipt.json) | 2026-09-26 07:45:30.346112; binds map, joint specification and protocol |
| Entire evidence inventory | [MANIFEST.json](../assets/ndfa_confirmation_20260926/MANIFEST.json) | SHA256 checks for 80 scientific files |

SHA256:

- Protocol: `b565f5490696b0afbd251c344599ee3f463a044298e8a8dbc3ff45464cb713f1`.
- Joint specification: `9b6880d5412a4681855d3b44742480113e08b29e1af550fa4959c244eb0484ae`.
- Interpretation map: `a50fbf7112906b2de58d7f5bd0fcbc0ed0b4e7a4fad91cb6ecaeb3840778d9cb`.

These are recorded project timestamps and integrity checks, not independent
third-party timestamp certification. The interpretation map is not described
as a preregistration preceding development. Earlier exposure to standard test
sets and the original MNIST cohort remain disclosed.

## Artifact map and public scope

| Purpose | Public artifact |
|---|---|
| All joint results and source | [880-run evidence](../assets/ndfa_confirmation_20260926/README.md), including eight `test_evaluation/summary.json` files and the 46-file source snapshot |
| Independent statistical reconstruction | [audit_joint_decision.py](../scripts/ndfa_strengthening_20260925/audit_joint_decision.py) |
| Original digit cohorts | [September 25 exports](../assets/ndfa_strengthening_20260925), explicitly separate from pooled extension summaries |
| Current historical figure inputs | [New input manifest](../assets/ndfa_arxiv_review_20260926/MANIFEST.json), plus the existing September 18/19 assets |
| Current historical figure renderer | [redraw.py](../scripts/ndfa_arxiv_review_20260926/redraw.py) and [mode_timing_plot.py](../scripts/ndfa_arxiv_review_20260926/mode_timing_plot.py) |
| Figure 5, confirmation tables and appendix | Manuscript `scripts/confirmation_material.py` and `data/joint_confirmation_20260926.json` |
| Publisher-deposited Dalm reference metadata | [Crossref record](../assets/ndfa_arxiv_review_20260926/dalm_publisher_metadata.json) |

Data archives, trained model checkpoints, and per-example prediction tensors
are not included. The public seed-level audit reconstructs statistics; it does
not independently infer predictions from model weights or verify wall-clock
measurements. Historical development records not listed in these manifests
remain archived with the project and are not implied to be public.

## Presentation changes

The revision distinguishes pseudo-gradients from optimizer steps, corrects
historical MNIST labels and stale references, identifies damping and SD/CI
conventions table by table, and distinguishes selected checkpoints from
complete-run costs. Figure 5 shows the prespecified one-sided noninferiority
bound; its other intervals are individual and unadjusted. Historical figure
labels are clearer, means remain unchanged, and no panels were removed.
The original qualitative abstract and arXiv omission of the AI statement are
retained at the authors' explicit request. The ICLR build retains its disclosure.

The last locally prepared ICLR PDF and its source were preserved separately.
Its identity with the actual OpenReview upload is not independently verified.
The newer arXiv build must not be mistaken for an updated conference submission.

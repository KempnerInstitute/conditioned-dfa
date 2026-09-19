# nDFA scientific revision — 18 September 2026

The external comments led to corrections, stronger controls and a focused extension of the original geometry-led paper. All 202 prospectively specified follow-up cases completed: 32 validation-only development cases and 170 final-test models. Every endpoint was independently recomputed from saved logits in float64, with source and artifact hashes checked. No seed, failed-looking finite endpoint, or test-selected checkpoint was substituted.

## What changed scientifically

The mode-timing gain now compares the minima of the complete risk functions, including signal bias and both noise terms. Exact conditional finite-sample risk is distinguished from the population-spectrum surrogate. The reviewer’s counterexample gives 10/21, and the mathematical tests check that value. Scalar output alone no longer stands in for alignment.

Synthetic uncertainty now resamples whole global-seed vectors over the fixed task grid. The point estimates are unchanged. Spatial experiments retain their complete crossed-design means without treating the 25 combinations as independent samples. The covariance-power table uses matched cohorts. The excluded historical K trace is removed.

The main methods now state hidden-weight norm matching, its zero convention and scope, and the exact sample-space implementation. Convolutional factor definitions match the actual estimators. Related work distinguishes algebraic rank from effective rank and acknowledges overlap with update orthogonalization and Shampoo. Canonical dataset/architecture references are added; the published Dalm citation is retained.

## Follow-up evidence

| Question | Result | Scope |
|---|---|---|
| Does A retain its gain with longer measured work? | A−DFA is +2.024 pp at 60 s and +2.258 pp at 120 s. | Five paired seeds per budget; earlier validation-selected settings transferred, with the schedule stretched rather than retuned. |
| Does it approach BP? | A−BP is −1.618 pp at 60 s and −0.612 pp at 120 s. | BP remains ahead; these budgets do not establish convergence. |
| Does it exceed forward decorrelation? | A−FD is −0.248 pp and +0.040 pp; both individual intervals include zero. | Neither superiority nor equivalence is established. |
| Does activity conditioning survive both tested widths? | A−DFA is +4.412 pp and +6.296 pp at 5,000 updates. | Separate five-seed confirmation at each width with shared BN. |
| Does width-specific E tuning resolve complementarity? | Full E selects the strongest searched damping at both widths; E is close to DFA and K is close to A. | The selected factor is nearly scalar. This suppresses the transformation rather than establishing a robust E gain. |
| Does full E outperform diagonal E? | The paired differences are small with intervals spanning zero. | Equal validation grids; strong full-E damping limits interpretation as a test of useful correlations. |
| What changes predictive stability? | BN restores ordinary losses across all four rules in the fixed recipe. Lower feedback scale without BN does not. | This isolates an effect under one learning-rate/damping recipe, not every cause of earlier protocol differences. |

The existing eight-seed activity control favors full over diagonal moments by 4.406 pp (individual paired-t interval [3.712, 5.100]). This is explicitly an additional analysis, outside the old declared comparison family.

The 50-model loss-tail audit confirms broad miscalibration in the unstable protocol. Extremely large median losses rule out an explanation based only on a handful of tail examples. The new stability intervention also shows extreme loss for raw DFA without BN, so this is not unique to inverse conditioning.

All follow-up intervals describe training-seed variation conditional on the split and protocol; five seeds provide limited information. They are individual, not simultaneous, intervals. The data remain CIFAR-10 and MLPs: neither ImageNet scaling nor general superiority to BP is established.

## Presentation and disposition

The practical matched-work results and absolute-accuracy table are now in the main paper. ImageNet block-output comparisons move to the supplement with their operator distinction preserved. Fourteen figures are redrawn, using consistent factor colors, larger panels, aligned external letters and corrected uncertainty. Three redundant historical figures are archived; the underlying scientific results remain available.

Removing forced page breaks reduces the full ICLR manuscript from 63 to 56 pages, without shrinking figures or deleting scientific content. All main figures and the main table fit within nine pages; the checker now enforces this explicitly.

The supplement retains theory, definitions, final endpoints, essential selection evidence, relevant limitations and the targeted follow-ups. New error-side diagonal experiments answer the selected mechanism question. A separate orthogonalized-DFA/Muon optimization study was optional in the assessment and was not added; its conceptual overlap is addressed directly. Checkpoint rotations were an alternative to the implemented diagonal ablation, not an additional claimed experiment.

The revision improves scientific reliability and makes the practical result stronger. ICLR competitiveness still rests on the explanatory contribution under approximate credit: inverse-moment preconditioning itself is established, and the demonstrated scale and task breadth remain limited. Acceptance should not be inferred from clean builds or the new gains.

## Evidence

- Frozen protocol: `docs/research/ndfa_revision_followups_protocol_20260918.md`
- Full new audit: `results/ndfa_revision_followups_audit_20260918/audit.json`
- All endpoint/contrast/spectrum tables: the CSVs in that audit directory
- Saved-data corrections: `results/ndfa_revision_saved_audit_20260918/`
- Corrected numerical simulation: `results/infodfa_mode_timing_corrected_20260918/`
- Figure generator: `scripts/ndfa_revision_20260918/redraw.py`
- Mathematical/cohort tests: 22 passed
- Slurm accounting: `docs/research/ndfa_revision_slurm_accounting_20260918.txt`

Jobs 47091857, 47091858 and 47091861 all completed on `kempner_h100_priority`, consuming 6,661 allocated GPU-seconds (about 1.85 H100-hours), within the new phase’s 220-minute ceiling. No external conference or arXiv submission has been performed.

## Verified release — 19 September

The final manuscript has nine scientific main pages including all five main figures and the main table, 56 total ICLR pages, 54 author-preprint pages, 17 figures and 59 references. Both builds have no unresolved references or overflowing boxes; active-figure text bounds show no clipping. The final main pages and full-document contact sheets were inspected. The arXiv source archive builds independently with identical text and rendered pages.

The compact anonymous review archive is 98,188,181 bytes. Fresh extraction passes integrity, retained numerical reconstructions and 143 CPU tests, all new endpoint checks, three mathematical tests, regeneration of all 13 figures handled by the revision generator, and the anonymity scan. The mode-timing simulator produces the fourteenth redrawn figure independently. The mathematical/cohort test suite also passes all 22 tests. Original and anonymized manifest hashes are checked separately; historical provenance is not overwritten.

The [artifact map](ndfa_revision_artifact_map_20260919.json) records exact hashes and availability. The manuscript commit is `7fa1eddfe59415e1e6979e0cfda039c4f49d66d1`; both repositories use `ndfa-revision-2026-09-19`. Code is released from the current public main branch with the revision's dependency closure, without incorporating unrelated unpublished development. The review ZIP and arXiv upload archives remain separate local deliverables, not public GitHub data downloads.

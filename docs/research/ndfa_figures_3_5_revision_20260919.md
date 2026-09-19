# Compact Figure 3 and expanded Figure 5 — 19 September 2026

Figure 3 is reduced from 2.55 to 1.80 inches in height at the final manuscript
width. All three panels, seed points, means, SEMs and predictive-loss values
remain. The repeated numerical mean annotations are removed. Panel letters
are aligned above and left of their panels, and tick labels are simplified.

Figure 5 now uses a four-panel row:

- A: the original five matched-work contrasts, grouped into the separate SGD
  and forward-decorrelation studies.
- B: the six existing longer-work contrasts, comparing activity conditioning
  with DFA, BP and forward-decorrelated DFA at both extended budgets.
- C: the original four K-minus-A contrasts with transferred damping, grouped
  by parameter count and distinguished by training-pool size.
- D: the six existing width-specific damping contrasts, showing the retained
  activity gain and the small incremental error-factor effect after selection.

Every plotted mean and individual 95% t interval is independently calculated
from paired values and checked against its saved source summary. The original
nine contrasts remain, including the two prespecified primary comparisons.
The twelve added contrasts already appeared in the supplementary follow-up
figure; there is no new training, selection, test inference or cohort pooling.
Eight-seed and five-seed cohorts are identified in the caption. Individual seed
points and interval endpoints are checked to remain inside the visible axes.

The supplementary follow-up figure now contains only the normalization and
predictive-loss intervention. Every endpoint and mean from that panel remains;
the work and width panels move to the main figure rather than being duplicated.
Main-text and supplementary cross-references are updated. Figures 1, 2 and 4
remain byte-identical to the preceding horizontal-layout revision.

The same compact generator now exports the five main figures and the
normalization figure. Figure 3's plotted numerical artists are checked against
the preceding generator, alongside the existing Figure 2/4 checks. The manifest
records all 21 Figure 5 contrasts, seed counts, intervals and primary flags.

Both manuscripts compile in three passes without unresolved references or
overfull boxes, with all main figures and tables inside the nine-page limit.
The ICLR manuscript remains 51 pages and the author preprint 49 pages. All 14
active figures pass vector-text bounds checks; the revised pages were visually
inspected. The new anonymous package preserves the experimental records byte
for byte and regenerates all six exported figure assets from a fresh extraction.
The arXiv package is separately rebuilt and compared in text and rendered pixels.

Current artifacts are mapped in
`ndfa_expanded_figures_artifact_map_20260919.json`. The source tag is
`ndfa-expanded-figures-2026-09-19` in both repositories. The preceding scientific
audits remain applicable; this pass changes presentation and evidence placement.

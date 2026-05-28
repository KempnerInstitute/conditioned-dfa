# Info-DFA ICLR runbook — addressing reviewer reject vectors

This runbook tracks the additions made to address the five reject-risk items
identified in review of `drafts/Info-DFA/conditioned_dfa_iclr.tex`, plus the
spatial-routing promotion and statistical-rigor pass.

All file paths are relative to the repository root.

## Status legend
- **DONE (no compute)**: text/script/figure already produced from existing CSVs.
- **DONE (script ready)**: script and slurm batch exist; needs cluster run.
- **PENDING**: needs further work.

---

## 1. Reject-risk #2 — nDFA and K-nDFA reported as separate methods
**Status:** DONE (no compute).

- Modified `analysis/write_infodfa_paper_tables.py` to compute and emit
  per-method `ndfa` and `kndfa` columns in both the synthetic and vision
  summaries.
- New tables written: `drafts/Info-DFA/tables/table_infodfa_synthetic_noise_split.tex`
  and `drafts/Info-DFA/tables/table_infodfa_vision_noise_split.tex`.
- Headline finding from the split: nDFA is the safe default; K-nDFA is the
  upgrade for the synthetic nuisance-dominant and mixed-context regimes
  (+2--3 pp over nDFA), and is slightly worse than nDFA on clean task-aligned
  and on the vision MLPs.
- **Paper action:** swap `\input{tables/table_infodfa_synthetic_noise.tex}` for
  `\input{tables/table_infodfa_synthetic_noise_split.tex}` (same for vision).
  Update narrative in §5.1 to reflect the (nDFA default, K-nDFA upgrade) story
  rather than "best of two per cell."

## 2. Reject-risk #3 — within-method Π-vs-accuracy diagnostic
**Status:** DONE (no compute).

- New script: `analysis/make_infodfa_pi_diagnostic.py`.
- Reads `results/infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_best_by_method.csv`
  and produces `drafts/Info-DFA/figures/iclr_fig_pi_diagnostic.{png,pdf,svg}`.
- New finding: between methods, Π and accuracy rise together (Π_nDFA ~3 vs
  Π_DFA ~0.04, mean acc ~62% vs ~49%). Within a single method the correlation
  inverts: nDFA r = −0.44, K-nDFA r = −0.48, DFA r = +0.79. Restricted-rank
  feedback (rank=1,2,4) brings nDFA/K-nDFA Π back toward DFA range and
  *increases* their accuracy from ~60% to >80% on the synthetic suite.
- **Paper action:** add a new §3.2 "Diagnostic prediction" referencing this
  figure; replace current Fig 2C "step > 0.1 vs step ≤ 0.1" interpretation
  with the cleaner between-vs-within distinction. Section text lives in the
  Discussion paragraph already drafted in
  `drafts/Info-DFA/sections/section_abstract_and_polish.tex`.

## 3. Reject-risk #5 — linearized analysis (theory anchor)
**Status:** DONE (no compute).

- New §3.1 written: `drafts/Info-DFA/sections/section_linearized_analysis.tex`.
- States the BP/DFA/nDFA expected updates explicitly, identifies the
  input-statistics weighting that conditioning removes, derives the gap
  $\Delta(\Sigma,\lambda_C)$, and proves a Cauchy-Schwarz-based inequality
  showing the gap is non-negative and grows with input anisotropy.
- **Paper action:** `\input{sections/section_linearized_analysis.tex}` after
  Eq. \ref{eq:kndfa}.

## 4. Reject-risk #1 — controls comparison (BP regularization + DFA + norm-match)
**Status:** DONE (script ready), NEEDS CLUSTER RUN.

- New experiment driver: `experiments/run_dfa_controls.py`.
- New slurm batch: `slurm/infodfa_controls.sbatch` (2 tasks: synthetic_nuisance
  and fashion_mnist_noisy, ~8h on 1 GPU).
- New aggregator: `analysis/write_infodfa_controls_table.py` produces
  `drafts/Info-DFA/tables/table_infodfa_controls.tex`.

To launch:
```bash
sbatch slurm/infodfa_controls.sbatch
# When complete:
python analysis/write_infodfa_controls_table.py
```

Smoke test (verified):
```bash
python experiments/run_dfa_controls.py \
    --output-dir results/dfa_controls_smoke \
    --n-seeds 1 --epochs 2 --hidden-dims 64 \
    --methods bp bp+l2 bp+label_smoothing bp+early_stop \
              dfa_random dfa_random+norm_match ndfa_random
```

Controls covered:
- BP + L2 (`--weight-decay 1e-3`)
- BP + label smoothing 0.1
- BP + early stop on val loss
- DFA + per-layer BP-norm matching (rescale each DFA layer-gradient to match
  the BP gradient norm on a held-out evaluation batch)
- nDFA / K-nDFA references

The deliverable is a single 5-column table comparing all methods at one
synthetic cell (nuisance-dominant) and one real-data cell
(Fashion-MNIST noisy) at matched (epochs, batch, lr, hidden_dims). This
single table closes the largest remaining reviewer attack surface.

## 5. Reject-risk #4 — NDFA on convnets explanation + spatial-Kronecker proposal
**Status:** DONE (no compute).

- New §6.1 / §6.2: `drafts/Info-DFA/sections/section_convnet_failure.tex`.
- Explains diagonal channel-NDFA's failure on ImageNet via two mechanisms
  (spatial covariance ignored; off-diagonal channel correlations biased).
- Proposes spatial-Kronecker variant $C^{\mathrm{ch}} \otimes C^{\mathrm{sp}}$
  (Eq. \ref{eq:spatial_kron}) as future work.
- **Paper action:** `\input{sections/section_convnet_failure.tex}` between
  current CIFAR-100 paragraph and ImageNet substitution-depth paragraph.

## 6. Spatial routing promotion (new main-paper §6.2)
**Status:** DONE (no compute).

- New §6.2: `drafts/Info-DFA/sections/section_spatial_routing.tex`.
- Uses the completed 90-epoch head-to-head aggregate at
  `results/imagenet100_infodfa_spatial_headtohead90_aggregate_20260526/imagenet_credit_assignment_summary.csv`.
- Includes a new main-paper table (`tab:infodfa_spatial_routing`) showing
  broadcast wins at every depth, bp_oracle and activation lose, and
  bp_sign_oracle is at random chance.
- **Paper action:** `\input{sections/section_spatial_routing.tex}` after the
  ImageNet substitution-depth subsection.

## 7. Refinetti alignment paragraph + expanded related work
**Status:** DONE (no compute), bib entries pending.

- New §7 (Related Work) replacement:
  `drafts/Info-DFA/sections/section_related_work_expanded.tex`.
- Adds Refinetti 2021 (align-then-memorize), Pogodin & Latham 2020 (Kolen-Pollack),
  Frenkel 2021 (PEPITA), Hinton 2022 (Forward-Forward), Crafton 2019 (DRTP).
- Includes a paragraph engaging directly with Refinetti and stating our
  testable prediction.
- **Paper action:** swap the existing `\section{Related Work}` for the
  expanded version, and add the four bibitems listed at the bottom of that
  file to the `\thebibliography` block.

## 8. Statistical rigor pass
**Status:** DONE (no compute).

- New script: `analysis/compute_infodfa_statistical_tests.py`.
- Output: `results/infodfa_paper_tables_20260527/infodfa_statistical_tests.csv`
  and `.md`.
- Paired t-tests and Wilcoxon signed-rank with Holm correction across the
  16 (regime, comparison) pairs (4 synthetic + 4 vision × 2 reference each).
- Key headline: nDFA vs DFA wins are p < 10^-7 across all four synthetic
  regimes; nDFA vs BP wins are p < 10^-13 in nuisance/mixed/low-sample;
  nDFA *loses* to BP on task-aligned (p = 5.6e-3) and on Fashion-MNIST is
  statistically tied (p = 0.51). The latter is the honest finding that
  required us to soften the abstract.
- **Paper action:** add an appendix \appref{app:stat_tests} that includes
  `infodfa_statistical_tests.md` rendered as a long table; cite it from §5.

## 9. Abstract / compute-cost / reproducibility polish
**Status:** DONE (no compute).

- All three live in `drafts/Info-DFA/sections/section_abstract_and_polish.tex`.
- Abstract is 210 words (down from 280), reports paired-test-backed numbers,
  acknowledges the task_aligned loss.
- Compute-cost paragraph estimates 3--12% wall-clock overhead.
- Reproducibility statement lists every script and slurm batch.
- New "When conditioning does not help" paragraph in Discussion.
- **Paper action:** copy the four blocks into the appropriate sections of
  `conditioned_dfa_iclr.tex`.

---

## Net effect on the paper

Main paper expands from approximately 7 pages to approximately 10 pages:

| Section | Before | After |
|---|---|---|
| Abstract | 280w | 210w (tighter, stat-backed) |
| §1 Introduction | unchanged | unchanged |
| §2 Conditioned DFA | unchanged | unchanged |
| §3 Diagnostics | 1 paragraph | **+ §3.1 linearized analysis (0.5pp)** |
| §3.2 Within-method Π | absent | **NEW (0.4pp + figure)** |
| §5 Experiments | unchanged | unchanged |
| §5.x Controls | absent | **NEW table + 0.4pp text** |
| §6 Convnets | 1 paragraph | **+ §6.1 diagonal-failure mechanism (0.5pp)** |
| §6.2 Spatial routing | half-paragraph | **NEW 0.6pp + new table** |
| §7 Related Work | 0.5pp | **1.0pp (Refinetti, KP, PEPITA, FF, DRTP)** |
| §7.x When conditioning fails | absent | **NEW 0.3pp** |
| §7.y Compute overhead | absent | **NEW 0.2pp** |
| Reproducibility | 1 sentence | **1 paragraph** |
| Appendix: stat tests | absent | **NEW long table** |

Net main-text addition: approximately 2.5--3.0 pages. Bib expansion: 4 new
entries. Two new tables in the main paper (controls + spatial routing).
One new main-paper figure (Π diagnostic) and one promoted supplementary
figure (spatial dynamics).

---

## Compute completed

- **Controls comparison** (`slurm/infodfa_controls.sbatch`, job 16640587):
  COMPLETED in 15 min. Synthetic-nuisance result: BP regularization moves <1pp,
  DFA+norm-match is *worse* than DFA, nDFA beats BP by +14pp. Fashion-MNIST:
  DFA+norm-match closes most of the gap (77.0%), tying nDFA (76.5%). Tables
  regenerated in `tables/table_infodfa_controls.tex` and merged into the paper.

## Compute completed: winner experiments

### Experiment A: ColoredMNIST (job 16647828, COMPLETED 2026-05-28)

**Outcome: DFA-rescue winner; included in paper.**

K-nDFA beats DFA on the held-out grayscale probe by $+11.9$\,pp ($p=0.032$,
paired $t$, 8 seeds); nDFA $+11.5$\,pp ($p=0.036$). Both also match or
marginally exceed BP. Norm-matching DFA does not exploit the conditioning
structure. Results in `results/dfa_coloredmnist_v1/`. Paper integration:
new $\S 5.3$ subsection, `tables/table_infodfa_coloredmnist.tex`, added
to abstract.

### Experiment B: low-data ImageNet-100 (job 16647831, COMPLETED 2026-05-28)

**Outcome: not a winner; dropped from paper.**

At 25 images/class, BP overfits (79\% train, 22\% val) but still beats
block-DFA by $+5.1$\,pp and block-NDFA by $+5.5$\,pp. NDFA \emph{underfits}
(train acc 67\% vs DFA's 72\%): the empirical covariance at this data scale
is sampling-noise-dominated, so the conditioning step over-whitens. This is
consistent with the within-method $\Pi$ finding (\S\ref{sec:pi_within_method}).
Results in `results/imagenet100_lowdata_v1/`; not added to manuscript per
"include only winners" policy.

### Experiment C: noisy-label ImageNet-100 (job 16647840, COMPLETED 2026-05-28)

**Outcome: not a winner; dropped from paper.**

All 9 cells completed in $\sim$50 minutes each (3 concurrent, total wall-clock
$\sim$3 hours, much faster than the 6-12h budgeted). 30\% symmetric label
noise, layer3+4 substitution, ResNet-18 on ImageNet-100.

Final val\_top1 (3 seeds each):
- BP: $70.27 \pm 0.24$ (overfits to noise: train\_top1 64.15)
- block-DFA: $62.66 \pm 0.41$ (underfits: train\_top1 49.78)
- block-NDFA: $62.59 \pm 0.33$ (underfits: train\_top1 49.80)

Paired t-tests on val\_top1:
- nDFA vs DFA: $-0.07$ pp, $p = 0.87$ (statistical tie)
- nDFA vs BP: $-7.68$ pp, $p = 0.005$ (significant loss to BP)
- DFA vs BP: $-7.61$ pp, $p = 0.006$ (significant loss to BP)

This confirms the existing convnet-NDFA failure finding (already reported in
\S 6.1 and Table 3): diagonal channel-NDFA on conv-net features is
indistinguishable from raw block-DFA across substitution depth, data scale,
and noise regime. Results in `results/imagenet100_noisy_v1/`. Aggregator at
`analysis/aggregate_imagenet100_noisy_v1.py`; it auto-checks the winner
threshold ($\Delta \ge 2$ pp $\wedge p < 0.05$) and writes a paper table only
if it passes. The threshold did not pass, so no paper edit was made.

## Final outcome: 1 winner out of 3 candidate experiments

- **A: ColoredMNIST** — winner; integrated into paper as \S 5.3 + new table.
- **B: low-data ImageNet-100** — not a winner; dropped.
- **C: noisy-label ImageNet-100** — not a winner; dropped.

The headline outcome of the "find places where conditioning beats BP on real
data" round is one clean real-image DFA-rescue (ColoredMNIST) and two
confirmations of the existing convnet failure mode (B and C). The paper's
21-page state is the final state.

After completion, aggregate with:
```bash
# Experiment A:
cat results/dfa_coloredmnist_v1/dfa_coloredmnist_summary.md
# Experiment B:
python analysis/aggregate_imagenet_credit_assignment.py \
    --output-dir results/imagenet100_lowdata_aggregate_v1 \
    --inputs 'results/imagenet100_lowdata_v1/**/imagenet_credit_assignment.csv'
# Experiment C:
python analysis/aggregate_imagenet_credit_assignment.py \
    --output-dir results/imagenet100_noisy_aggregate_v1 \
    --inputs 'results/imagenet100_noisy_v1/**/imagenet_credit_assignment.csv'
```

## Compute still pending

(none)

---

## Suggested compile order

1. Apply abstract, repro, and discussion-paragraph edits from
   `sections/section_abstract_and_polish.tex` to
   `conditioned_dfa_iclr.tex`.
2. Add `\input{sections/section_linearized_analysis.tex}` after Eq. \ref{eq:kndfa}.
3. Swap the related-work block with
   `sections/section_related_work_expanded.tex` and add the four bib entries.
4. Add `\input{sections/section_convnet_failure.tex}` and
   `\input{sections/section_spatial_routing.tex}` in §6.
5. Add the new Π-diagnostic figure (`iclr_fig_pi_diagnostic.pdf`) and a short
   §3.2 referencing it.
6. Launch the controls slurm batch, wait, regenerate the controls table.
7. Swap the legacy synthetic and vision tables for the new split versions.
8. Add the statistical-tests appendix.

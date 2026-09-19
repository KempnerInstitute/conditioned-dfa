# Horizontal figure revision — 19 September 2026

Figures 1, 2 and 4 return to four-panel rows, designed at the final 5.5-inch
manuscript width. The previous two-row versions used excessive vertical space.
The revised figure canvases save approximately four inches in the main text,
without shrinking an oversized source figure to fit the page. Panel letters
share fixed row coordinates above and to the left of the panels. Legends are
outside data where the scatter or bars would otherwise be obscured.

Figure 1 again follows geometry, spectral weights, conditioning bound and
realized learning behavior. Its restored third panel evaluates the analytic
input-condition-number identity; it is labeled analytic rather than simulated.
The mixed residual in the schematic, corrected theoretical scope and recovered
population simulation protocol remain. The update formulas previously drawn in
panel C are already defined in the method section. Figure 2 and Figure 4 retain
all numerical curves, bars, points and uncertainty estimates from the corrected
preceding generator. The new script asserts their identity before export and
separately checks Figure 1's spectral weights and mean simulation update counts.

The original archived duplicate kernel-patch panel is not reinstated because
the dedicated spatial figure presents that experiment. Other secondary archived
diagnostics remain available with their original data and source history. No
experimental data or result was removed in this presentation revision.

The top-float limit now permits two floats, including a figure/table pair.
This moves Figures 4 and 5 from pages 8 and 9 to pages 7 and 8, nearer their
discussion. The main text still fits within nine pages; the overall lengths
remain 51 pages for ICLR and 49 for the author preprint. The active inventory
remains 14 figures, 49 references and 13 supplementary sections. The abstract
retains the author's requested qualitative comparisons.

Both official-style builds pass three compilations with no overfull boxes or
unresolved references. Figure text-bound checks find no clipping. Main pages
were inspected as rendered pages; figure and supplementary float ownership
checks pass. The revised anonymous archive contains byte-identical experimental
records. Its final figure regeneration and independently rebuilt paper are
checked from a fresh extraction outside the workspace. The arXiv package is
also rebuilt independently and compared in full text and pixels.

Use the September 19 figure generator, then
`scripts/ndfa_figure_layout_20260919/redraw.py` for the final three compact
overlays. Detailed commands are in `REPRODUCE.md` and the anonymous archive's
`revision/HORIZONTAL_FIGURES.md`. This revision does not rerun training or test
inference. Artifact locations and hashes are recorded in
`ndfa_horizontal_figures_artifact_map_20260919.json`.

"""Restyle the damping-law validation figure from the saved CSVs (no re-simulation).

Reads ONLY the aggregates written by analysis/validate_damping_theory.py:
  results/infodfa_damping_theory_v1/damping_theory_curves.csv
  results/infodfa_damping_theory_v1/damping_theory_optima.csv
  results/infodfa_damping_theory_v1/damping_theory_fits.csv

and redraws the three validation panels in the paper's house style
(drafts/Info-DFA/scripts/make_iclr_figures.py conventions):
  A  realized kappa(M-hat) vs damping lambda_C by n/d at d=32, kappa=100
     (interior optimum; dotted verticals at lambda_C*)
  B  lambda_C* vs d/n, log-log, with the fitted-slope line (paper quotes 1.17)
     and the perturbative 1/4 lower-bound rate
  C  lambda_C* vs kappa(Sigma), log-log, with the predicted -1/2 slope

The slopes shown are verified against damping_theory_fits.csv: the CSV stores
per-(d,kappa) and per-(d,n/d) slopes, so the displayed aggregate is refit from
the optima CSV and asserted to match the mean of the stored slopes to 2
decimals. Outputs damping_theory_validation.{pdf,png,svg} into both
results/infodfa_damping_theory_v1/ and drafts/Info-DFA/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe (cluster nodes have no display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "infodfa_damping_theory_v1"
DRAFT_FIGURES = REPO / "drafts" / "Info-DFA" / "figures"

# House style: mirrors the rcParams block of
# drafts/Info-DFA/scripts/make_iclr_figures.py (figures drawn at ~6.4in width
# and rendered at \textwidth, so a 9pt label prints at ~7.7pt).
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.2,
        "lines.markersize": 4.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.handlelength": 1.7,
        "legend.columnspacing": 1.3,
        "legend.borderaxespad": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 400,
    }
)

GRID_KW = dict(color="#E8EAE6", linewidth=0.6)
NOD_GRID = [2, 4, 8, 16, 32, 64]
# Colorblind-safe sequential ramp (perceptually uniform viridis, truncated so
# the lightest step still reads on white) for the n/d family; shared between
# panels A and C so a given n/d keeps its color across panels.
NOD_COLORS = {nod: plt.cm.viridis(t) for nod, t in zip(NOD_GRID, np.linspace(0.02, 0.80, 6))}
KAPPA_COLORS = {10.0: "#9ECAE1", 50.0: "#4292C6", 100.0: "#08519C"}  # single-hue blues
D_MARKERS = {32: dict(marker="o", mfc=None, ls="-"), 128: dict(marker="s", mfc="none", ls=(0, (4, 2)))}


def label_axes(axes, fontsize=10, pad_x_pt=1.0, pad_y_pt=2.5):
    """Bold corner panel letters outside each panel (house convention).

    Simplified from make_iclr_figures.label_axes for a single-row figure:
    letters sit just above/left of each panel's tight bounding box and share
    one baseline across the row.
    """
    axes = list(axes)
    fig = axes[0].figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    corners = [inv.transform((ax.get_tightbbox(renderer).x0, ax.get_tightbbox(renderer).y1)) for ax in axes]
    top = max(cy for _, cy in corners)
    fig_w, fig_h = fig.get_size_inches()
    dx = pad_x_pt / 72.0 / fig_w
    dy = pad_y_pt / 72.0 / fig_h
    for letter, (cx, _) in zip("ABC", corners):
        fig.text(max(cx - dx, 0.0), top + dy, letter, fontsize=fontsize,
                 fontweight="bold", ha="left", va="bottom")


def loglog_slope(x, y) -> float:
    return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])


def anchored_powerlaw(ax, x_pts, y_pts, slope, shift=1.0, span=(None, None), **plot_kw):
    """Draw y = c * x^slope through the log-centroid of (x_pts, y_pts).

    `shift` multiplies the anchor height, e.g. to park a reference-slope line
    in the gap between point families instead of on top of one."""
    lx, ly = np.log(np.asarray(x_pts, float)), np.log(np.asarray(y_pts, float))
    x0, y0 = lx.mean(), ly.mean() + np.log(shift)
    xmin = span[0] if span[0] is not None else float(np.exp(lx.min()))
    xmax = span[1] if span[1] is not None else float(np.exp(lx.max()))
    xs = np.geomspace(xmin, xmax, 32)
    ax.plot(xs, np.exp(y0 + slope * (np.log(xs) - x0)), **plot_kw)


def verify_slopes(fits: pd.DataFrame, opt: pd.DataFrame) -> tuple[float, float]:
    """Aggregate slopes shown on the figure, verified against the fits CSV.

    The CSV stores the slope per (d, kappa) group (panel B) and per (d, n/d)
    group (panel C); the figure quotes one aggregate slope per panel, so we
    refit it from the optima CSV and assert it matches the mean of the stored
    group slopes to 2 decimals. The paper (Table + caption) quotes 1.17 and
    -0.52; note 1.17 truncates the pooled 1.1753.
    """
    stored_b = fits.loc[fits["fit"].eq("lam_star_vs_d_over_n"), "slope_cond"]
    stored_c = fits.loc[fits["fit"].eq("lam_star_vs_kappa"), "slope_cond"]
    refit_b = loglog_slope(opt["d_over_n"], opt["lam_star_cond"])
    refit_c = loglog_slope(opt["kappa"], opt["lam_star_cond"])
    assert abs(refit_b - stored_b.mean()) < 5e-3, (
        f"panel-B refit {refit_b:.4f} != mean stored slope {stored_b.mean():.4f}")
    assert abs(refit_c - stored_c.mean()) < 5e-3, (
        f"panel-C refit {refit_c:.4f} != mean stored slope {stored_c.mean():.4f}")
    # Displayed values are the paper-quoted 1.17 / -0.52.
    assert abs(refit_b - 1.17) < 6e-3, f"panel-B slope {refit_b:.4f} drifted from quoted 1.17"
    assert abs(refit_c - (-0.52)) < 5e-3, f"panel-C slope {refit_c:.4f} drifted from quoted -0.52"
    print(f"slope check: B refit {refit_b:.4f} vs stored mean {stored_b.mean():.4f} (shown 1.17); "
          f"C refit {refit_c:.4f} vs stored mean {stored_c.mean():.4f} (shown -0.52)")
    return refit_b, refit_c


def main() -> None:
    curves = pd.read_csv(RESULTS / "damping_theory_curves.csv")
    opt = pd.read_csv(RESULTS / "damping_theory_optima.csv")
    fits = pd.read_csv(RESULTS / "damping_theory_fits.csv")
    slope_b, _ = verify_slopes(fits, opt)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(6.4, 2.3), constrained_layout=True)

    # A: realized condition number vs damping has an interior optimum that
    # moves to heavier damping as n/d shrinks (d=32, kappa=100 slice).
    d0, k0 = 32, 100.0
    for nod in NOD_GRID:
        color = NOD_COLORS[nod]
        sub = curves[(curves.d == d0) & (curves.kappa == k0)
                     & (curves.n_over_d == nod) & (curves.metric == "cond")].sort_values("lam")
        axA.plot(sub.lam, sub.value, "-", color=color, lw=1.3, marker="o", ms=1.9,
                 markeredgewidth=0, label=f"{nod}")
        so = opt[(opt.d == d0) & (opt.kappa == k0) & (opt.n_over_d == nod)]
        axA.axvline(float(so.lam_star_cond.iloc[0]), color=color, ls=":", lw=0.8, alpha=0.85)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel(r"damping $\lambda_C$")
    axA.set_ylabel(r"realized $\kappa(\widehat M)$")
    axA.set_title("Interior optimum", fontsize=8.5)
    axA.text(0.03, 0.04, r"$d{=}32$, $\kappa{=}100$", transform=axA.transAxes,
             fontsize=5.8, va="bottom", ha="left")
    axA.legend(frameon=False, fontsize=5.2, ncol=2, loc="upper right",
               title=r"$n/d$", title_fontsize=5.8, handlelength=1.2,
               borderaxespad=0.2, columnspacing=0.8, labelspacing=0.35)
    axA.grid(which="major", **GRID_KW)
    axA.set_axisbelow(True)

    # B: the optimum grows with d/n. Color encodes kappa (sequential blues),
    # marker/linestyle encode d; the solid line is the pooled fit (slope
    # 1.1753, quoted as 1.17 in the paper), the gray dashed line the
    # perturbative 1/4 lower-bound rate, both through the log-centroid.
    for d, mk in D_MARKERS.items():
        for kappa, color in KAPPA_COLORS.items():
            sub = opt[(opt.d == d) & (opt.kappa == kappa)].sort_values("d_over_n")
            axB.plot(sub.d_over_n, sub.lam_star_cond, ls=mk["ls"], marker=mk["marker"],
                     mfc=mk["mfc"], color=color, lw=1.0, ms=2.6, mew=0.8)
    anchored_powerlaw(axB, opt.d_over_n, opt.lam_star_cond, slope_b,
                      color="#333333", lw=1.1, ls="-", zorder=1)
    anchored_powerlaw(axB, opt.d_over_n, opt.lam_star_cond, 0.25,
                      color="#999999", lw=1.0, ls=(0, (4, 2)), zorder=1)
    handles = [
        *(Line2D([], [], color=c, lw=1.6, label=rf"$\kappa{{=}}{k:.0f}$")
          for k, c in KAPPA_COLORS.items()),
        Line2D([], [], color="#333333", lw=1.1, label="fit 1.17"),
        Line2D([], [], color="#555555", lw=0, marker="o", ms=2.8, label=r"$d{=}32$"),
        Line2D([], [], color="#555555", lw=0, marker="s", ms=2.8, mfc="none", label=r"$d{=}128$"),
        Line2D([], [], color="#999999", lw=1.0, ls=(0, (4, 2)), label="1/4 bound"),
    ]
    axB.legend(handles=handles, frameon=False, fontsize=5.2, loc="upper left",
               ncol=2, handlelength=1.5, borderaxespad=0.2, columnspacing=0.8,
               labelspacing=0.35)
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel(r"$d/n$")
    axB.set_ylabel(r"$\lambda_C^\star$")
    axB.set_title("Sample-size rate", fontsize=8.5)
    # ~1 decade of headroom above the data for the in-panel legend
    axB.set_ylim(top=float(opt.lam_star_cond.max()) * 12.0)
    axB.grid(which="major", **GRID_KW)
    axB.set_axisbelow(True)

    # C: the optimum follows the predicted spectral law
    # lambda_C* ~ sqrt(lam_min lam_max) ~ kappa^{-1/2} (fit -0.52). Same n/d
    # ramp as panel A (subset for legibility); marker/linestyle encode d.
    nod_subset = [2, 8, 32]
    for d, mk in D_MARKERS.items():
        for nod in nod_subset:
            sub = opt[(opt.d == d) & (opt.n_over_d == nod)].sort_values("kappa")
            axC.plot(sub.kappa, sub.lam_star_cond, ls=mk["ls"], marker=mk["marker"],
                     mfc=mk["mfc"], color=NOD_COLORS[nod], lw=1.0, ms=2.6, mew=0.8)
    subset = opt[opt.n_over_d.isin(nod_subset)]
    # shift=3 parks the reference-slope line in the gap between the n/d=2 and
    # n/d=8 families so it cannot be misread as a d=128 (dashed) series.
    anchored_powerlaw(axC, subset.kappa, subset.lam_star_cond, -0.5, shift=3.0,
                      color="#222222", lw=1.1, ls=(0, (4, 2)), zorder=1)
    handles = [
        *(Line2D([], [], color=NOD_COLORS[nod], lw=1.6, label=rf"$n/d{{=}}{nod}$")
          for nod in nod_subset),
        Line2D([], [], color="#555555", lw=0, marker="o", ms=2.8, label=r"$d{=}32$"),
        Line2D([], [], color="#555555", lw=0, marker="s", ms=2.8, mfc="none", label=r"$d{=}128$"),
        Line2D([], [], color="#222222", lw=1.1, ls=(0, (4, 2)), label=r"$-1/2$ pred"),
    ]
    axC.legend(handles=handles, frameon=False, fontsize=5.2, loc="lower left",
               ncol=2, handlelength=1.5, borderaxespad=0.2, columnspacing=0.8,
               labelspacing=0.35)
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.set_xlabel(r"$\kappa(\Sigma)$")
    axC.set_ylabel(r"$\lambda_C^\star$")
    axC.set_title("Spectral law", fontsize=8.5)
    axC.text(0.96, 0.96, r"fit $-0.52$", transform=axC.transAxes, fontsize=5.8,
             va="top", ha="right")
    axC.grid(which="major", **GRID_KW)
    axC.set_axisbelow(True)

    label_axes([axA, axB, axC])
    for outdir in (RESULTS, DRAFT_FIGURES):
        outdir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(outdir / f"damping_theory_validation.{ext}", bbox_inches="tight", dpi=400)
        print("wrote", outdir / "damping_theory_validation.{pdf,png,svg}")
    plt.close(fig)


if __name__ == "__main__":
    main()

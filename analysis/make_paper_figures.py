"""Generate publication-quality paper figures and schematics.

The script reads result CSVs under ``results/`` and writes vector PDFs/SVGs plus
high-resolution PNGs under ``results/paper_figures``. When draft checkouts are
present, figures are mirrored into ``drafts/Info-Man/figures`` and
``drafts/Info-DFA/figures`` for Overleaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "paper_figures"
INFO_MAN_OUT = OUT / "Info-Man"
INFO_DFA_OUT = OUT / "Info-DFA"
SHARED_OUT = OUT / "shared"
INFO_MAN_DRAFT = ROOT / "drafts" / "Info-Man" / "figures"
INFO_DFA_DRAFT = ROOT / "drafts" / "Info-DFA" / "figures"
SHARED_DRAFT = ROOT / "drafts" / "figures"


COLORS = {
    "total": "#4C78A8",
    "used": "#59A14F",
    "behavior": "#F28E2B",
    "pid": "#B07AA1",
    "null": "#9D9D9D",
    "bad": "#E15759",
    "stim": "#4C78A8",
    "choice": "#F28E2B",
    "comm": "#76B7B2",
    "dfa": "#59A14F",
    "bp": "#4C78A8",
    "ndfa": "#B07AA1",
}


GROUP_COLORS = {
    "visual": "#4C78A8",
    "somatosensory": "#72B7B2",
    "frontal_motor": "#59A14F",
    "basal_ganglia": "#B07AA1",
    "midbrain": "#F28E2B",
    "thalamus": "#ECA82C",
    "hippocampal": "#9D755D",
    "other": "#9D9D9D",
}


METHOD_ORDER = [
    "bp",
    "dfa_bp_aligned",
    "dfa_bp_aligned_dynamic",
    "dfa_random",
    "ndfa_random",
    "ndfa_random_error",
    "ndfa_random_kronecker",
    "drtp_random",
    "dfa_tangent_biased",
    "dfa_tangent_orthogonal",
    "dfa_negative_bp",
]


LABELS = {
    "total_mi_geom": "Total MI",
    "used_mi_geom": "Used MI",
    "used_total_ratio": "Used / total",
    "pid_intersection": "PID intersection",
    "projection_mi_gcmi_used": "GCMI projection",
    "projection_mi_knn_used": "kNN projection",
    "total_geom": "Total geometric",
    "used_geom_decoder": "Projected geometric",
    "projection_binned_decoder": "Binned projection",
    "projection_gcmi_decoder": "Gaussian-copula MI",
    "projection_knn_decoder": "kNN projection",
    "pid_imin_decoder": "PID $I_{min}$",
    "sender_total_mi": "Sender total",
    "sender_comm_mi": "Communication",
    "receiver_used_mi": "Receiver used",
    "triadic_transferred_used_mi": "Triadic transferred",
    "triadic_calibrated_mi": "Calibrated transfer",
    "receiver_prediction_r2": "Receiver prediction",
    "early_param_cosine": "Parameter cosine",
    "early_activity_cosine": "Activity cosine",
    "early_tangent_cosine": "Tangent cosine",
    "early_task_tangent_cosine": "Task-tangent cosine",
    "early_fisher_trace": "Whitened Fisher",
    "delta_fisher_trace_early": "Fisher growth",
    "early_weight_rayleigh": "Rayleigh quotient",
    "delta_weight_rayleigh_early": "Rayleigh growth",
    "early_class_dprime2": "Class d-prime",
    "delta_class_dprime2_early": "Class d-prime growth",
}


@dataclass(frozen=True)
class FigureRecord:
    path: Path
    title: str
    paper: str


def main() -> None:
    setup_style()
    for path in (INFO_MAN_OUT, INFO_DFA_OUT, SHARED_OUT, INFO_MAN_DRAFT, INFO_DFA_DRAFT, SHARED_DRAFT):
        path.mkdir(parents=True, exist_ok=True)

    records: list[FigureRecord] = []
    # Info-DFA-only: Info-Man builders are retained as dead code pending
    # a post-publication cleanup pass.
    records.extend(make_infodfa_figures())
    records.extend(make_shared_dashboards())
    write_manifest(records)
    print(f"Saved {len(records)} figure sets to {OUT}")
    print(f"Mirrored paper figures to {INFO_DFA_DRAFT}")


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_infoman_figures() -> list[FigureRecord]:
    records = []
    records.append(save_figure(infoman_schematic(), "infoman_fig1_schematic", INFO_MAN_OUT, INFO_MAN_DRAFT, "Info-Man conceptual schematic", "Info-Man"))
    records.append(save_figure(infoman_core_results(), "infoman_fig2_core_results", INFO_MAN_OUT, INFO_MAN_DRAFT, "Info-Man core synthetic and estimator results", "Info-Man"))
    records.append(save_figure(infoman_extensions(), "infoman_fig3_extensions", INFO_MAN_OUT, INFO_MAN_DRAFT, "Info-Man communication and mixed-selectivity extensions", "Info-Man"))
    records.append(save_figure(infoman_real_data(), "infoman_fig4_real_data", INFO_MAN_OUT, INFO_MAN_DRAFT, "Info-Man real-data analysis suite", "Info-Man"))
    records.append(save_figure(infoman_real_communication_robustness(), "infoman_fig5_real_communication_robustness", INFO_MAN_OUT, INFO_MAN_DRAFT, "Info-Man real communication robustness", "Info-Man"))
    records.append(save_figure(infoman_openalyx_validation(), "infoman_fig6_openalyx_validation", INFO_MAN_OUT, INFO_MAN_DRAFT, "Info-Man independent OpenAlyx validation", "Info-Man"))
    records.append(save_figure(infoman_followup_diagnostics(), "infoman_supp_followup_diagnostics", INFO_MAN_OUT, INFO_MAN_DRAFT, "Info-Man follow-up diagnostics", "Info-Man"))
    records.append(save_figure(infoman_all_results_dashboard(), "infoman_supp_all_results_dashboard", INFO_MAN_OUT, INFO_MAN_DRAFT, "Info-Man all-results dashboard", "Info-Man"))
    return records


def make_infodfa_figures() -> list[FigureRecord]:
    records = []
    records.append(save_figure(infodfa_schematic(), "infodfa_fig1_schematic", INFO_DFA_OUT, INFO_DFA_DRAFT, "Info-DFA conceptual schematic", "Info-DFA"))
    records.append(save_figure(infodfa_synthetic_results(), "infodfa_fig2_synthetic_results", INFO_DFA_OUT, INFO_DFA_DRAFT, "Info-DFA synthetic manifold results", "Info-DFA"))
    records.append(save_figure(infodfa_vision_results(), "infodfa_fig3_vision_bridge", INFO_DFA_OUT, INFO_DFA_DRAFT, "Info-DFA vision bridge", "Info-DFA"))
    records.append(save_figure(infodfa_all_results_dashboard(), "infodfa_supp_all_results_dashboard", INFO_DFA_OUT, INFO_DFA_DRAFT, "Info-DFA all-results dashboard", "Info-DFA"))
    return records


def make_shared_dashboards() -> list[FigureRecord]:
    fig = figure_index_panel()
    return [save_figure(fig, "paper_figure_index", SHARED_OUT, SHARED_DRAFT, "Paper figure index", "shared")]


def infoman_schematic() -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.set_title("Whiten by uncertainty")
    rng = np.random.default_rng(1)
    pts = rng.multivariate_normal([0, 0], [[2.2, 1.1], [1.1, 0.9]], size=160)
    ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.25, color=COLORS["null"], edgecolor="none")
    ellipse_arrow(ax, (-2.2, -1.6), (2.3, 1.6), COLORS["stim"], "raw responses")
    ax.arrow(3.0, 0.0, 0.9, 0.0, head_width=0.16, head_length=0.18, color="black", length_includes_head=True)
    pts_w = pts @ np.linalg.inv(np.linalg.cholesky(np.cov(pts.T))).T
    ax.scatter(pts_w[:, 0] + 5.0, pts_w[:, 1], s=6, alpha=0.25, color=COLORS["used"], edgecolor="none")
    ellipse_arrow(ax, (4.2, -1.1), (5.8, 1.1), COLORS["used"], "whitened")
    ax.set_xlim(-3.2, 6.5)
    ax.set_ylim(-3, 3)
    ax.axis("off")

    ax = axes[0, 1]
    ax.set_title("Project onto the behaviorally relevant subspace")
    draw_axes(ax)
    ax.arrow(0, 0, 1.8, 1.0, head_width=0.08, head_length=0.12, color=COLORS["stim"], length_includes_head=True)
    ax.arrow(0, 0, 2.1, 0.0, head_width=0.08, head_length=0.12, color=COLORS["choice"], length_includes_head=True)
    ax.plot([1.8, 1.8], [0, 1.0], linestyle="--", color=COLORS["null"], linewidth=1)
    ax.arrow(0, 0, 1.8, 0, head_width=0.06, head_length=0.1, color=COLORS["used"], length_includes_head=True)
    ax.text(1.0, 1.0, "total", color=COLORS["stim"], ha="center")
    ax.text(1.1, -0.25, "used", color=COLORS["used"], ha="center")
    ax.text(1.8, 0.12, "choice axis", color=COLORS["choice"], ha="center")
    ax.set_xlim(-0.2, 2.6)
    ax.set_ylim(-0.55, 1.55)
    ax.axis("off")

    ax = axes[1, 0]
    ax.set_title("Intersection information on held-out trials")
    draw_node(ax, (0.1, 0.55), "Stimulus\nS", COLORS["stim"])
    draw_node(ax, (0.5, 0.55), "Neural\nprojection", COLORS["used"])
    draw_node(ax, (0.9, 0.55), "Choice\nC", COLORS["choice"])
    arrow_between(ax, (0.24, 0.55), (0.36, 0.55))
    arrow_between(ax, (0.64, 0.55), (0.76, 0.55))
    ax.text(0.5, 0.18, r"$I_\cap(S;R_{\hat c};C)$", ha="center", fontsize=12)
    ax.text(0.5, 0.02, "axis estimated on train, information evaluated on test", ha="center", fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax = axes[1, 1]
    ax.set_title("Communication + choice geometry")
    draw_node(ax, (0.16, 0.68), "Sender\npopulation", COLORS["stim"])
    draw_node(ax, (0.52, 0.68), "Communication\nsubspace", COLORS["comm"])
    draw_node(ax, (0.86, 0.68), "Receiver\nchoice geometry", COLORS["choice"])
    arrow_between(ax, (0.29, 0.68), (0.39, 0.68))
    arrow_between(ax, (0.65, 0.68), (0.73, 0.68))
    ax.text(0.52, 0.28, "triadic transferred used information", ha="center", fontsize=9)
    ax.text(0.52, 0.13, "stimulus geometry ∩ communication ∩ receiver choice", ha="center", fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_labels(axes.ravel())
    return fig


def infoman_core_results() -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.6), constrained_layout=True)
    pop = read_csv("population_geometry/population_geometry_results.csv")
    ctx = read_csv("population_context/population_context_results.csv")
    scoreboard = read_scoreboard_results()
    trial = read_csv("population_trial_sweep/population_trial_sweep_results.csv")

    ax = axes[0, 0]
    if not pop.empty:
        grouped = pop.groupby("angle_deg").mean(numeric_only=True).reset_index()
        ax.plot(grouped["angle_deg"], grouped["total_mi_geom"], color=COLORS["total"], marker="o", label="total")
        ax.plot(grouped["angle_deg"], grouped["used_mi_geom_known_axis"], color=COLORS["used"], marker="o", label="used")
        ax.plot(grouped["angle_deg"], grouped["behavior_mi"], color=COLORS["behavior"], marker="o", label="behavior")
    ax.set_xlabel("Stimulus-choice angle (deg)")
    ax.set_ylabel("Information (bits)")
    ax.set_title("Axis rotation benchmark")
    legend_if_handles(ax)

    ax = axes[0, 1]
    if not ctx.empty:
        by_role = ctx.groupby("role").mean(numeric_only=True).reset_index()
        metrics = ["total_mi_geom", "used_mi_geom", "pid_intersection_used", "behavior_mi"]
        labels = ["total", "held-out used", "PID", "behavior"]
        colors = [COLORS["total"], COLORS["used"], COLORS["pid"], COLORS["behavior"]]
        grouped_bars(ax, by_role, "role", metrics, labels, colors)
    ax.set_ylabel("Information (bits)")
    ax.set_title("Context relevance without oracle axes")

    ax = axes[1, 0]
    if not scoreboard.empty:
        score_summary = (
            scoreboard.groupby(["scenario", "estimator"])["abs_error_behavior"].mean().reset_index()
        )
        plot_scoreboard_heatmap(ax, score_summary)
    ax.set_title("Estimator scoreboard")

    ax = axes[1, 1]
    trial_col = "trial_count" if "trial_count" in trial.columns else "n_trials" if "n_trials" in trial.columns else None
    if not trial.empty and trial_col is not None:
        trial_summary = trial.groupby(trial_col).mean(numeric_only=True).reset_index()
        for metric, color, label in [
            ("cv_behavior_mi", COLORS["behavior"], "behavior"),
            ("cv_total_mi_geom", COLORS["total"], "total"),
            ("cv_used_mi_decoder", COLORS["used"], "decoder used"),
            ("cv_pid_intersection_decoder", COLORS["pid"], "CV PID"),
        ]:
            if metric in trial_summary:
                ax.plot(trial_summary[trial_col], trial_summary[metric], marker="o", label=label, color=color)
        ax.set_xscale("log")
    else:
        add_no_data(ax, "trial sweep")
    ax.set_xlabel("Trials")
    ax.set_ylabel("Information (bits)")
    ax.set_title("Finite-sample convergence")
    legend_if_handles(ax)

    fig.suptitle("Info-Man: projected geometry estimates behaviorally used information", fontsize=11.5)
    add_panel_labels(axes.ravel())
    return fig


def infoman_extensions() -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.6), constrained_layout=True)
    comm = read_first_csv(
        "infoman_communication_subspace_slurm_v4/infoman_communication_subspace.csv",
        "infoman_communication_subspace/infoman_communication_subspace.csv",
    )
    comm_scores = read_first_csv(
        "infoman_communication_subspace_slurm_v4/infoman_communication_predictor_scores.csv",
        "infoman_communication_subspace/infoman_communication_predictor_scores.csv",
    )
    mixed = read_first_csv(
        "infoman_mixed_selectivity_slurm_v4/infoman_mixed_selectivity.csv",
        "infoman_mixed_selectivity/infoman_mixed_selectivity.csv",
    )
    mixed_scores = read_first_csv(
        "infoman_mixed_selectivity_slurm_v4/infoman_mixed_selectivity_predictor_scores.csv",
        "infoman_mixed_selectivity/infoman_mixed_selectivity_predictor_scores.csv",
    )

    ax = axes[0, 0]
    if not comm.empty:
        by_angle = comm.groupby("comm_angle_deg").mean(numeric_only=True).reset_index()
        for metric, color in [
            ("behavior_mi_bits", COLORS["behavior"]),
            ("sender_total_mi", COLORS["total"]),
            ("sender_comm_mi", COLORS["comm"]),
            ("triadic_transferred_used_mi", COLORS["used"]),
        ]:
            ax.plot(by_angle["comm_angle_deg"], by_angle[metric], marker="o", color=color, label=LABELS.get(metric, metric))
    ax.set_xlabel("Sender stimulus/communication angle (deg)")
    ax.set_ylabel("Information (bits)")
    ax.set_title("Communication-subspace benchmark")
    legend_if_handles(ax, fontsize=6.8)

    ax = axes[0, 1]
    if not comm_scores.empty:
        plot_predictor_bars(ax, comm_scores, color=COLORS["comm"], label_map=LABELS)
    ax.set_title("Communication predictors")

    ax = axes[1, 0]
    if not mixed.empty:
        by_alpha = mixed.groupby("alpha").mean(numeric_only=True).reset_index()
        ax2 = ax.twinx()
        ax.plot(by_alpha["alpha"], by_alpha["kappa_mix"], marker="o", color=COLORS["used"], label=r"$\kappa_{mix}$")
        ax2.plot(by_alpha["alpha"], by_alpha["synergy_imin"], marker="o", color=COLORS["pid"], label="PID synergy")
        ax.set_ylabel("Whitened mixed curvature", color=COLORS["used"])
        ax2.set_ylabel("Synergy (bits)", color=COLORS["pid"])
        ax.set_xlabel("Mixed-selectivity amplitude")
        legend_if_handles(ax, loc="upper left")
        legend_if_handles(ax2, loc="lower right")
    ax.set_title("Mixed selectivity predicts synergy")

    ax = axes[1, 1]
    if not mixed_scores.empty:
        plot_predictor_bars(ax, mixed_scores, color=COLORS["pid"], label_map=LABELS)
    ax.set_title("Synergy predictors")

    fig.suptitle("Info-Man: beyond single choice-axis geometry", fontsize=11.5)
    add_panel_labels(axes.ravel())
    return fig


def infoman_real_data() -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(8.9, 5.7), constrained_layout=True)
    area = normalize_area_summary(
        read_first_csv(
            "steinmetz_real_aggregate_slurm_v4/aggregate_area_session_summary.csv",
            "aggregate_real_neural/aggregate_area_session_summary.csv",
        )
    )
    if area.empty:
        area = normalize_area_summary(
            read_first_csv(
                "steinmetz_real_aggregate_slurm_v4/aggregate_area_summary.csv",
                "aggregate_real_neural/aggregate_area_summary.csv",
            )
        )
    timecourse = read_first_csv(
        "steinmetz_real_aggregate_slurm_v4/aggregate_timecourse_summary.csv",
        "aggregate_real_neural/aggregate_timecourse_summary.csv",
    )
    confounds = read_first_csv(
        "steinmetz_real_aggregate_slurm_v4/aggregate_confound_summary.csv",
        "aggregate_real_neural/aggregate_confound_summary.csv",
    )
    comm = read_first_csv(
        "steinmetz_real_aggregate_slurm_v4/aggregate_communication_summary.csv",
        "aggregate_real_neural_confirmatory/aggregate_communication_summary.csv",
    )
    comm_rank = read_first_csv(
        "steinmetz_real_aggregate_slurm_v4/aggregate_communication_rank_summary.csv",
        "aggregate_real_neural_confirmatory/aggregate_communication_rank_summary.csv",
    )
    if comm.empty:
        comm = read_csv("aggregate_real_neural/aggregate_communication_summary.csv")
        comm_rank = read_csv("aggregate_real_neural/aggregate_communication_rank_summary.csv")
    plot_area = select_real_areas(area, max_areas=12)
    plot_areas = set(plot_area["area"]) if not plot_area.empty else set()

    ax = axes[0, 0]
    if not plot_area.empty:
        plot_df = plot_area.sort_values("used_total_ratio_mean", ascending=True)
        y = np.arange(plot_df.shape[0])
        group_colors = [GROUP_COLORS.get(g, GROUP_COLORS["other"]) for g in plot_df.get("area_group", pd.Series(["other"] * len(plot_df)))]
        ax.barh(y, plot_df["total_mi_mean"], color="0.82", alpha=0.9, label="total")
        ax.barh(y, plot_df["used_mi_mean"], color=group_colors, alpha=0.9, label="used")
        ax.set_yticks(y, plot_df["area"])
    ax.set_xlabel("Information (bits)")
    ax.set_title("Area-level total and used information")
    legend_if_handles(ax)

    ax = axes[0, 1]
    if not plot_area.empty:
        plot_df = plot_area.sort_values("used_total_ratio_mean", ascending=True)
        y = np.arange(plot_df.shape[0])
        group_colors = [GROUP_COLORS.get(g, GROUP_COLORS["other"]) for g in plot_df.get("area_group", pd.Series(["other"] * len(plot_df)))]
        ax.barh(y, plot_df["used_total_ratio_mean"], color=group_colors)
        if {"used_total_ratio_ci_low", "used_total_ratio_ci_high"}.issubset(plot_df.columns):
            xerr = np.vstack(
                [
                    plot_df["used_total_ratio_mean"] - plot_df["used_total_ratio_ci_low"],
                    plot_df["used_total_ratio_ci_high"] - plot_df["used_total_ratio_mean"],
                ]
            )
            ax.errorbar(plot_df["used_total_ratio_mean"], y, xerr=xerr, fmt="none", color="black", linewidth=0.8, capsize=2)
        ax.set_yticks(y, plot_df["area"])
    ax.set_xlabel("Used / total information")
    ax.set_title("Behavioral-use fraction")
    add_group_legend(ax, plot_area)

    ax = axes[0, 2]
    if not plot_area.empty and "pid_intersection_mean" in plot_area.columns:
        plot_df = plot_area.sort_values("pid_intersection_mean", ascending=True)
        ax.barh(plot_df["area"], plot_df["pid_intersection_mean"], color=COLORS["pid"])
    ax.set_xlabel("PID intersection (bits)")
    ax.set_title("Area-level intersection")

    ax = axes[1, 0]
    if not timecourse.empty:
        time_plot = timecourse[timecourse["area"].isin(plot_areas)] if plot_areas else timecourse
        top = time_plot.groupby("area")["used_mi_geom"].mean().sort_values(ascending=False).head(5).index
        for area_name in top:
            sub = time_plot[time_plot["area"] == area_name].sort_values("time_bin_center")
            ax.plot(sub["time_bin_center"], sub["used_mi_geom"], marker="o", label=area_name)
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Used MI (bits)")
    ax.set_title("Time-resolved used information")
    legend_if_handles(ax, fontsize=7)

    ax = axes[1, 1]
    if not confounds.empty:
        residual = confounds[confounds["variant"] != "raw"].copy()
        if plot_areas:
            residual = residual[residual["area"].isin(plot_areas)]
        residual = residual[residual["raw_used_mi"] > 0.02].sort_values("used_mi_retained_fraction", ascending=True)
        residual["retained_plot"] = residual["used_mi_retained_fraction"].clip(upper=2.0)
        ax.barh(residual["area"], residual["retained_plot"], color=COLORS["bad"])
        ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(2.0, color="0.65", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Retained fraction (clipped at 2)")
    ax.set_title("Confound residualization")

    ax = axes[1, 2]
    if not comm.empty:
        comm_metric = "triadic_calibrated_mi" if "triadic_calibrated_mi" in comm.columns else "triadic_transferred_used_mi"
        qc_comm = comm.copy()
        if {"receiver_prediction_r2", "triadic_p"}.issubset(qc_comm.columns):
            qc_comm = qc_comm[(qc_comm["receiver_prediction_r2"] > 0.0) & (qc_comm["triadic_p"] < 0.10)]
        if qc_comm.empty:
            qc_comm = comm
        top_comm = qc_comm.sort_values(comm_metric, ascending=False).head(6).sort_values(comm_metric)
        labels = [
            f"{s}->{r} r{int(rank)}"
            for s, r, rank in zip(top_comm["sender_area"], top_comm["receiver_area"], top_comm["comm_rank"])
        ]
        colors = [
            COLORS["comm"] if ("triadic_p" not in top_comm or p < 0.05) else mpl.colors.to_rgba(COLORS["comm"], 0.45)
            for p in top_comm.get("triadic_p", pd.Series(np.zeros(top_comm.shape[0])))
        ]
        ax.barh(labels, top_comm[comm_metric], color=colors)
        if not comm_rank.empty and "comm_rank" in comm_rank:
            best = comm_rank.iloc[0]
            ax.text(
                0.98,
                0.04,
                f"best rank: {int(best['comm_rank'])}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.2,
            )
    else:
        add_no_data(ax, "communication")
    ax.set_xlabel("Calibrated transfer (bits)")
    ax.set_title("Communication screen candidates")

    fig.suptitle("Info-Man: real-data analysis suite", fontsize=11.5)
    add_panel_labels(axes.ravel())
    return fig


def normalize_area_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    renames = {
        "total_mi_session_mean": "total_mi_mean",
        "used_mi_session_mean": "used_mi_mean",
        "used_total_ratio_session_mean": "used_total_ratio_mean",
        "used_total_ratio_session_ci_low": "used_total_ratio_ci_low",
        "used_total_ratio_session_ci_high": "used_total_ratio_ci_high",
        "pid_intersection_session_mean": "pid_intersection_mean",
    }
    out = out.rename(columns={k: v for k, v in renames.items() if k in out.columns})
    if "area_group" not in out.columns and "area" in out.columns:
        out["area_group"] = "other"
    return out


def add_group_legend(ax, df: pd.DataFrame) -> None:
    if df.empty or "area_group" not in df.columns:
        return
    groups = [g for g in GROUP_COLORS if g in set(df["area_group"].dropna())]
    handles = [mpl.patches.Patch(facecolor=GROUP_COLORS[g], edgecolor="none", label=g.replace("_", "/")) for g in groups]
    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=6.3, loc="lower right")


def infoman_real_communication_robustness() -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(8.9, 5.4), constrained_layout=True)
    df = read_first_csv(
        "steinmetz_real_communication_robustness/communication_robustness.csv",
        "steinmetz_real_communication_robustness/communication_robustness.csv",
    )
    ibl_comm = read_csv("ibl_openalyx_aggregate_v4/aggregate_communication_summary.csv")
    stz_comm = read_csv("steinmetz_real_aggregate_slurm_v4/aggregate_communication_summary.csv")

    ax = axes[0, 0]
    ax.set_title("Calibration logic")
    draw_node(ax, (0.12, 0.7), "Raw\nprojection", COLORS["comm"])
    draw_node(ax, (0.43, 0.7), "Intersection\nbounds", COLORS["pid"])
    draw_node(ax, (0.74, 0.7), "Choice-stratified\nnull", COLORS["null"])
    draw_node(ax, (0.43, 0.24), "Calibrated\ntransfer", COLORS["used"])
    arrow_between(ax, (0.24, 0.7), (0.31, 0.7))
    arrow_between(ax, (0.55, 0.7), (0.62, 0.7))
    arrow_between(ax, (0.74, 0.58), (0.52, 0.32), color=COLORS["null"])
    arrow_between(ax, (0.43, 0.58), (0.43, 0.36), color=COLORS["pid"])
    ax.text(0.43, 0.05, r"$\max(0, I_\mathrm{bound}-I_\mathrm{null})$", ha="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax = axes[0, 1]
    if not df.empty:
        rank_df = df[df["comparison"] == "rank_confirmatory"].copy()
        for pair, sub in rank_df.groupby("pair"):
            sub = sub.sort_values("comm_rank")
            ax.plot(sub["comm_rank"], sub["triadic_calibrated_mi"], marker="o", label=pair)
        if not rank_df.empty:
            ax.set_xticks(sorted(rank_df["comm_rank"].unique()))
    elif not ibl_comm.empty:
        top = ibl_comm.sort_values("triadic_calibrated_mi", ascending=False).head(8)
        for pair, sub in top.groupby(["sender_area", "receiver_area"]):
            sub = sub.sort_values("comm_rank")
            ax.plot(sub["comm_rank"], sub["triadic_calibrated_mi"], marker="o", label=f"{pair[0]}->{pair[1]}")
    ax.set_xlabel("Communication rank")
    ax.set_ylabel("Calibrated transfer (bits)")
    ax.set_title("Screening rank sensitivity")
    legend_if_handles(ax, fontsize=7.5)

    ax = axes[0, 2]
    if not df.empty:
        specificity = df[(df["comparison"] == "direction_and_pair_control") & (df["comm_rank"] == 1)].copy()
        if not specificity.empty:
            specificity = specificity.sort_values("triadic_calibrated_mi", ascending=True)
            target_pairs = {"IC->ACA", "IC->MRN"}
            colors = []
            for _, row in specificity.iterrows():
                robust = row["pair"] in target_pairs and row["triadic_p"] < 0.05 and row["receiver_prediction_r2"] > 0.01
                exploratory = row["triadic_p"] < 0.05 and row["receiver_prediction_r2"] > 0.0
                colors.append(COLORS["comm"] if robust else (mpl.colors.to_rgba(COLORS["behavior"], 0.65) if exploratory else "0.78"))
            ax.barh(specificity["pair"], specificity["triadic_calibrated_mi"], color=colors)
            ax.text(
                0.98,
                0.04,
                "teal: target QC\norange: exploratory/control",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.5,
                color="0.35",
            )
    else:
        top = ibl_comm.copy()
        if not top.empty:
            qc = (top["triadic_p"] < 0.05) & (top["receiver_prediction_r2"] > 0.0)
            loose = (top["triadic_p"] < 0.10) & (top["receiver_prediction_r2"] > 0.0)
            top = top.assign(qc=qc, loose=loose).sort_values("triadic_calibrated_mi", ascending=False).head(10)
            labels = [f"{s}->{r} r{int(k)}" for s, r, k in zip(top["sender_area"], top["receiver_area"], top["comm_rank"])]
            colors = [COLORS["comm"] if q else (mpl.colors.to_rgba(COLORS["behavior"], 0.65) if l else "0.78") for q, l in zip(top["qc"], top["loose"])]
            ax.barh(labels[::-1], top["triadic_calibrated_mi"].to_numpy()[::-1], color=colors[::-1])
            ax.text(0.98, 0.04, "teal: p<.05 & R2>0\norange: p<.10 & R2>0", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="0.35")
    ax.set_xlabel("Calibrated transfer (bits)")
    ax.set_title("OpenAlyx communication screen")

    ax = axes[1, 0]
    if not df.empty:
        time_df = df[(df["comparison"] == "time_window") & (df["comm_rank"] == 1)]
        for pair, sub in time_df.groupby("pair"):
            sub = sub.sort_values("time_bin_center")
            ax.plot(sub["time_bin_center"], sub["triadic_calibrated_mi"], marker="o", label=pair)
    elif not stz_comm.empty:
        top = stz_comm.sort_values("triadic_calibrated_mi", ascending=False).head(8).sort_values("triadic_calibrated_mi")
        labels = [f"{s}->{r} r{int(k)}" for s, r, k in zip(top["sender_area"], top["receiver_area"], top["comm_rank"])]
        ax.barh(labels, top["triadic_calibrated_mi"], color="0.72")
    ax.set_xlabel("Time bin center")
    ax.set_ylabel("Calibrated transfer (bits)")
    ax.set_title("Steinmetz screen did not pass QC")
    legend_if_handles(ax, fontsize=7.5)

    ax = axes[1, 1]
    if not df.empty:
        unit_df = df[(df["comparison"] == "unit_count") & (df["comm_rank"] == 1)]
        if not unit_df.empty:
            unit_summary = unit_df.groupby(["pair", "max_units"])["triadic_calibrated_mi"].agg(["mean", "sem"]).reset_index()
            for pair, sub in unit_summary.groupby("pair"):
                sub = sub.sort_values("max_units")
                ax.errorbar(sub["max_units"], sub["mean"], yerr=sub["sem"].fillna(0.0), marker="o", capsize=2, label=pair)
    else:
        counts = []
        for name, table in [("OpenAlyx", ibl_comm), ("Steinmetz", stz_comm)]:
            if table.empty:
                continue
            counts.append(
                {
                    "dataset": name,
                    "p<.10": float(((table["triadic_p"] < 0.10) & (table["receiver_prediction_r2"] > 0.0)).sum()),
                    "p<.05": float(((table["triadic_p"] < 0.05) & (table["receiver_prediction_r2"] > 0.0)).sum()),
                }
            )
        if counts:
            count_df = pd.DataFrame(counts).set_index("dataset")
            x = np.arange(count_df.shape[0])
            ax.bar(x - 0.18, count_df["p<.10"], width=0.36, label="p<.10 & R2>0", color=COLORS["behavior"])
            ax.bar(x + 0.18, count_df["p<.05"], width=0.36, label="p<.05 & R2>0", color=COLORS["comm"])
            ax.set_xticks(x, count_df.index)
    ax.set_xlabel("Maximum units per area")
    ax.set_ylabel("Candidate count")
    ax.set_title("QC threshold sensitivity")
    legend_if_handles(ax, fontsize=7.5)

    ax = axes[1, 2]
    if not df.empty:
        qc = df[df["comparison"].isin(["rank_confirmatory", "direction_and_pair_control"])].copy()
        if not qc.empty:
            is_sig = (qc["triadic_p"] < 0.05) & (qc["receiver_prediction_r2"] > 0.0)
            ax.scatter(
                qc.loc[~is_sig, "receiver_prediction_r2"],
                qc.loc[~is_sig, "triadic_calibrated_mi"],
                s=28,
                color="0.75",
                label="control / weak",
            )
            ax.scatter(
                qc.loc[is_sig, "receiver_prediction_r2"],
                qc.loc[is_sig, "triadic_calibrated_mi"],
                s=34,
                color=COLORS["comm"],
                label="QC-passing",
            )
            seen_pairs = set()
            for _, row in qc.sort_values("triadic_calibrated_mi", ascending=False).iterrows():
                if row["pair"] in seen_pairs:
                    continue
                ax.text(row["receiver_prediction_r2"], row["triadic_calibrated_mi"], row["pair"], fontsize=6.5, ha="left", va="bottom")
                seen_pairs.add(row["pair"])
                if len(seen_pairs) >= 4:
                    break
    else:
        qc = pd.concat(
            [
                ibl_comm.assign(dataset="OpenAlyx"),
                stz_comm.assign(dataset="Steinmetz"),
            ],
            ignore_index=True,
            sort=False,
        )
        if not qc.empty:
            is_sig = (qc["triadic_p"] < 0.05) & (qc["receiver_prediction_r2"] > 0.0)
            ax.scatter(qc.loc[~is_sig, "receiver_prediction_r2"], qc.loc[~is_sig, "triadic_calibrated_mi"], s=16, color="0.78", alpha=0.55, label="screened")
            ax.scatter(qc.loc[is_sig, "receiver_prediction_r2"], qc.loc[is_sig, "triadic_calibrated_mi"], s=30, color=COLORS["comm"], label="p<.05 & R2>0")
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Receiver prediction R2")
    ax.set_ylabel("Calibrated transfer (bits)")
    ax.set_title("QC gate")
    legend_if_handles(ax, fontsize=7)

    fig.suptitle("Info-Man: calibrated real communication screens remain exploratory", fontsize=11.5)
    add_panel_labels(axes.ravel())
    return fig


def infoman_openalyx_validation() -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(8.9, 2.8), constrained_layout=True)
    df = read_first_csv(
        "ibl_openalyx_aggregate_v4/aggregate_area_session_summary.csv",
        "ibl_openalyx_aggregate_v4/aggregate_area_summary.csv",
        "ibl_openalyx_area_batch_aggregate/aggregate_area_summary.csv",
    )
    if df.empty:
        df = read_csv("ibl_openalyx_smoke_session/real_neural_sweep.csv")

    ax = axes[0]
    if not df.empty:
        summary = summarize_ibl_area_sweep(df).sort_values("used_total_ratio", ascending=True)
        y = np.arange(summary.shape[0])
        ax.barh(y, summary["total_mi_geom"], color="0.82", label="total")
        ax.barh(y, summary["used_mi_geom"], color=COLORS["used"], label="used")
        ax.set_yticks(y, summary["area"])
    else:
        add_no_data(ax, "OpenAlyx sweep")
    ax.set_xlabel("Information (bits)")
    ax.set_title("Held-out total vs used")
    legend_if_handles(ax)

    ax = axes[1]
    if not df.empty:
        summary = summarize_ibl_area_sweep(df).sort_values("used_total_ratio", ascending=True)
        y = np.arange(summary.shape[0])
        ax.barh(y, summary["used_total_ratio"], color=COLORS["comm"])
        if {"used_total_ratio_low", "used_total_ratio_high"}.issubset(summary.columns):
            xerr = np.vstack(
                [
                    summary["used_total_ratio"] - summary["used_total_ratio_low"],
                    summary["used_total_ratio_high"] - summary["used_total_ratio"],
                ]
            )
            ax.errorbar(summary["used_total_ratio"], y, xerr=xerr, fmt="none", color="black", linewidth=0.8, capsize=2)
        elif "used_total_ratio_sem" in summary:
            ax.errorbar(
                summary["used_total_ratio"],
                y,
                xerr=summary["used_total_ratio_sem"].fillna(0.0),
                fmt="none",
                color="black",
                linewidth=0.8,
                capsize=2,
            )
        ax.set_yticks(y, summary["area"])
    ax.set_xlabel("Used / total")
    ax.set_title("Independent-data use fraction")

    ax = axes[2]
    if not df.empty:
        summary = summarize_ibl_area_sweep(df)
        ax.scatter(
            summary["total_mi_geom"],
            summary["used_mi_geom"],
            s=42,
            color=COLORS["used"],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
        for _, row in summary.iterrows():
            ax.text(row["total_mi_geom"], row["used_mi_geom"], row["area"], fontsize=7, ha="left", va="bottom")
        max_val = max(summary["total_mi_geom"].max(), summary["used_mi_geom"].max()) * 1.05
        ax.plot([0, max_val], [0, max_val], color="0.65", linestyle="--", linewidth=0.8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Total MI (bits)")
    ax.set_ylabel("Used MI (bits)")
    ax.set_title("Used information is a small subset")

    fig.suptitle("Info-Man: independent OpenAlyx validation cohort", fontsize=11.5)
    add_panel_labels(axes)
    return fig


def summarize_ibl_area_sweep(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["total_mi_geom", "used_mi_geom", "used_total_ratio", "pid_intersection", "choice_axis_acc"]
    if {"total_mi_session_mean", "used_mi_session_mean", "used_total_ratio_session_mean"}.issubset(df.columns):
        out = df.rename(
            columns={
                "total_mi_session_mean": "total_mi_geom",
                "used_mi_session_mean": "used_mi_geom",
                "used_total_ratio_session_mean": "used_total_ratio",
                "used_total_ratio_session_ci_low": "used_total_ratio_low",
                "used_total_ratio_session_ci_high": "used_total_ratio_high",
                "pid_intersection_session_mean": "pid_intersection",
            }
        )
        keep = [col for col in ["area", *metrics, "used_total_ratio_low", "used_total_ratio_high", "n_sessions"] if col in out.columns]
        return out[keep].sort_values("used_total_ratio", ascending=False)
    if {"total_mi_mean", "used_mi_mean", "used_total_ratio_mean"}.issubset(df.columns):
        out = df.rename(
            columns={
                "total_mi_mean": "total_mi_geom",
                "used_mi_mean": "used_mi_geom",
                "used_total_ratio_mean": "used_total_ratio",
                "used_total_ratio_ci_low": "used_total_ratio_low",
                "used_total_ratio_ci_high": "used_total_ratio_high",
                "pid_intersection_mean": "pid_intersection",
            }
        )
        keep = [col for col in ["area", *metrics, "used_total_ratio_low", "used_total_ratio_high", "n_sessions"] if col in out.columns]
        return out[keep].sort_values("used_total_ratio", ascending=False)
    summary = df.groupby("area")[metrics].agg(["mean", "sem"]).reset_index()
    summary.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary = summary.rename(columns={f"{metric}_mean": metric for metric in metrics})
    return summary.sort_values("used_total_ratio", ascending=False)


def infoman_all_results_dashboard() -> plt.Figure:
    fig, axes = plt.subplots(3, 3, figsize=(9.0, 7.2), constrained_layout=True)
    axes = axes.ravel()
    datasets = [
        ("Population geometry", "population_geometry/population_predictor_scores.csv"),
        ("Context task", "population_context/population_context_predictor_scores.csv"),
        ("Trial sweep", "population_trial_sweep/population_trial_sweep_predictor_scores.csv"),
        ("Estimator scoreboard", "infoman_estimator_scoreboard_slurm_v4/axis_rotation/infoman_estimator_scoreboard.csv"),
        ("Communication", "infoman_communication_subspace_slurm_v4/infoman_communication_predictor_scores.csv"),
        ("Mixed selectivity", "infoman_mixed_selectivity_slurm_v4/infoman_mixed_selectivity_predictor_scores.csv"),
        ("Real area", "steinmetz_real_aggregate_slurm_v4/aggregate_area_summary.csv"),
        ("Real confounds", "steinmetz_real_aggregate_slurm_v4/aggregate_confound_summary.csv"),
        ("Real communication", "ibl_openalyx_aggregate_v4/aggregate_communication_summary.csv"),
    ]
    for ax, (title, rel) in zip(axes, datasets):
        df = read_csv(rel)
        plot_generic_result(ax, title, df)
    fig.suptitle("Info-Man supplementary dashboard: current result coverage", fontsize=11.5)
    add_panel_labels(axes)
    return fig


def infoman_followup_diagnostics() -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), constrained_layout=True)

    ax = axes[0]
    pica = read_csv("infoman_pica_comparison_v1/infoman_pica_comparison.csv")
    if pica.empty:
        add_no_data(ax, "Pica-style comparison")
    else:
        order = pica.groupby("estimator")["abs_error_truth"].mean().sort_values()
        colors = [COLORS["bad"] if est == "geom_total" else COLORS["pid"] if "pica" in est else COLORS["used"] for est in order.index]
        ax.barh([pretty_estimator(est) for est in order.index], order.values, color=colors)
        ax.set_xlabel("Mean absolute error to ground truth")
        ax.set_title(r"Pica-style $I_\cap$ comparison")

    ax = axes[1]
    trial_frames = []
    for source, rel in [
        ("Steinmetz", "steinmetz_real_trial_sweep_aggregate_v1/real_neural_trial_sweep_all.csv"),
        ("OpenAlyx", "ibl_real_trial_sweep_aggregate_v1/real_neural_trial_sweep_all.csv"),
    ]:
        df = read_csv(rel)
        if not df.empty:
            df = df.copy()
            df["dataset_label"] = source
            trial_frames.append(df)
    if not trial_frames:
        add_no_data(ax, "Real-data trial convergence")
    else:
        trials = pd.concat(trial_frames, ignore_index=True, sort=False)
        for source, color in [("Steinmetz", COLORS["choice"]), ("OpenAlyx", COLORS["comm"])]:
            subset = trials[trials["dataset_label"] == source]
            if subset.empty:
                continue
            curve = subset.groupby("trial_count")["abs_error_used_vs_full"].mean().sort_index()
            ax.plot(curve.index, curve.values, marker="o", linewidth=1.5, label=source, color=color)
        ax.set_xscale("log")
        ax.set_xlabel("Trials")
        ax.set_ylabel("Absolute error vs full session")
        ax.set_title("Held-out used MI converges with trials")
        legend_if_handles(ax)

    ax = axes[2]
    critic = read_csv("infoman_neural_critic_calibration_v1/infoman_neural_critic_calibration.csv")
    if critic.empty:
        add_no_data(ax, "Neural critic calibration")
    else:
        neural = critic[critic["estimator"].astype(str).str.startswith("neural_")].copy()
        if neural.empty:
            add_no_data(ax, "Neural critic calibration")
        else:
            for estimator, group in neural.groupby("estimator"):
                curve = group.groupby("critic_steps")["abs_error_theoretical_used"].mean().sort_index()
                ax.plot(curve.index, curve.values, marker="o", linewidth=1.3, label=pretty_estimator(estimator))
            ax.set_xscale("log")
            ax.set_xlabel("Critic training steps")
            ax.set_ylabel("Mean absolute error")
            ax.set_title("Do neural critics calibrate?")
            legend_if_handles(ax, fontsize=6.2)

    fig.suptitle("Info-Man: defensive follow-up diagnostics", fontsize=11.2)
    add_panel_labels(axes)
    return fig


def select_real_areas(area: pd.DataFrame, *, max_areas: int) -> pd.DataFrame:
    if area.empty:
        return area
    df = area.copy()
    if "n_sessions" in df.columns:
        stable = df[df["n_sessions"] >= 2]
        if stable.shape[0] >= min(max_areas, 4):
            df = stable
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["used_total_ratio_mean"])
    return df.sort_values(["used_total_ratio_mean", "n_sessions" if "n_sessions" in df.columns else "area"], ascending=[False, False]).head(max_areas)


def infodfa_schematic() -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.set_title("BP versus DFA")
    draw_network(ax, mode="dfa")
    ax.text(0.18, 0.08, "forward weights", color=COLORS["bp"], ha="center")
    ax.text(0.72, 0.08, "fixed/random feedback", color=COLORS["dfa"], ha="center")
    ax.axis("off")

    ax = axes[0, 1]
    ax.set_title("Task tangent versus normal")
    t = np.linspace(-2.2, 2.2, 120)
    ax.plot(t, 0.35 * t**2 - 0.8, color="black", linewidth=1.2)
    point = np.array([0.3, 0.35 * 0.3**2 - 0.8])
    ax.scatter([point[0]], [point[1]], color="black", s=20, zorder=3)
    ax.arrow(point[0], point[1], 1.0, 0.25, color=COLORS["used"], head_width=0.08, length_includes_head=True)
    ax.arrow(point[0], point[1], -0.25, 1.0, color=COLORS["bad"], head_width=0.08, length_includes_head=True)
    ax.text(1.05, -0.5, "task tangent", color=COLORS["used"])
    ax.text(-0.5, 0.55, "normal", color=COLORS["bad"])
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-1.2, 1.6)
    ax.axis("off")

    ax = axes[1, 0]
    ax.set_title("Noise-whitened Fisher geometry")
    draw_axes(ax)
    ax.arrow(0, 0, 1.6, 0.4, color=COLORS["stim"], head_width=0.08, length_includes_head=True)
    ax.arrow(0, 0, 0.7, 1.4, color=COLORS["pid"], head_width=0.08, length_includes_head=True)
    ax.text(1.1, 0.05, r"$J$", color=COLORS["stim"])
    ax.text(0.35, 1.25, r"$\Sigma^{-1/2}J$", color=COLORS["pid"])
    ax.text(0.55, -0.38, r"$\mathrm{tr}(J^\top \Sigma^{-1}J)$", ha="center", fontsize=10)
    ax.set_xlim(-0.2, 2.1)
    ax.set_ylim(-0.65, 1.9)
    ax.axis("off")

    ax = axes[1, 1]
    ax.set_title("Rank and conditioning predictions")
    ranks = np.arange(1, 6)
    perf = 1 - np.exp(-0.8 * ranks)
    ax.plot(ranks, perf, marker="o", color=COLORS["dfa"], label="DFA succeeds")
    ax.axvline(2.0, color=COLORS["null"], linestyle="--", linewidth=1, label="task tangent rank")
    ax.set_xlabel("Feedback rank")
    ax.set_ylabel("Predicted accuracy")
    ax.set_ylim(0, 1.05)
    legend_if_handles(ax)
    add_panel_labels(axes.ravel())
    return fig


def infodfa_synthetic_results() -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.6), constrained_layout=True)
    df = read_first_csv("dfa_synthetic_full_slurm_v5/dfa_synthetic_results.csv", "dfa_synthetic_multimanifold/dfa_synthetic_results.csv", "dfa_synthetic/dfa_synthetic_results.csv")
    summary = read_first_csv("dfa_synthetic_full_slurm_v5/dfa_run_summary.csv", "dfa_synthetic_multimanifold/dfa_run_summary.csv", "dfa_synthetic/dfa_run_summary.csv")
    scores = read_first_csv("dfa_synthetic_full_slurm_v5/dfa_predictor_scores.csv", "dfa_synthetic_multimanifold/dfa_predictor_scores.csv", "dfa_synthetic/dfa_predictor_scores.csv")

    ax = axes[0, 0]
    if not df.empty:
        acc_df = df[["method", "epoch", "test_acc", "seed", "feedback_seed", "feedback_scale", "manifold"]].drop_duplicates()
        for method in sorted(acc_df["method"].unique(), key=method_sort_key):
            group = acc_df[acc_df["method"] == method]
            curve = group.groupby("epoch")["test_acc"].mean()
            ax.plot(curve.index, curve.values, marker="o", linewidth=1.4, label=pretty_method(method), color=method_color(method))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Learning curves")
    legend_if_handles(ax, fontsize=6.5)

    ax = axes[0, 1]
    if not df.empty:
        layer1 = df[(df["layer"] == 1) & (df["method"] != "bp")]
        for method in sorted(layer1["method"].unique(), key=method_sort_key):
            group = layer1[layer1["method"] == method]
            curve = group.groupby("epoch")["task_tangent_cosine"].mean()
            ax.plot(curve.index, curve.values, marker="o", linewidth=1.4, label=pretty_method(method), color=method_color(method))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Task-tangent cosine")
    ax.set_title("Alignment in task geometry")

    ax = axes[1, 0]
    if not summary.empty:
        dfa_runs = summary[summary["method"] != "bp"]
        for metric, color, label in [
            ("early_task_tangent_cosine", COLORS["dfa"], "task tangent"),
            ("early_fisher_trace", COLORS["pid"], "Fisher"),
            ("early_weight_rayleigh", COLORS["bad"], "Rayleigh"),
        ]:
            if metric not in dfa_runs:
                continue
            values = dfa_runs[metric].to_numpy(dtype=float)
            values = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-12)
            ax.scatter(values, dfa_runs["final_test_acc"], s=28, color=color, alpha=0.62, label=label)
    ax.set_xlabel("Early predictor (z score)")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Early diagnostics predict learning")
    legend_if_handles(ax)

    ax = axes[1, 1]
    if not scores.empty:
        keep = [
            "early_param_cosine",
            "early_tangent_cosine",
            "early_task_tangent_cosine",
            "early_fisher_trace",
            "early_weight_rayleigh",
            "delta_class_dprime2_early",
        ]
        plot_predictor_bars(ax, scores[scores["predictor"].isin(keep)], color=COLORS["dfa"], label_map=LABELS)
    ax.set_title("Predictor comparison")

    fig.suptitle("Info-DFA: tangent/Fisher diagnostics of local learning", fontsize=11.5)
    add_panel_labels(axes.ravel())
    return fig


def infodfa_vision_results() -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), constrained_layout=True)
    df = read_vision_results()

    ax = axes[0]
    if not df.empty:
        for method in sorted(df["method"].unique(), key=method_sort_key):
            group = df[df["method"] == method]
            curve = group.groupby("epoch")["test_acc"].mean()
            ax.plot(curve.index, curve.values, marker="o", linewidth=1.4, label=pretty_method(method), color=method_color(method))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Vision learning curves")
    legend_if_handles(ax, fontsize=6.5)

    ax = axes[1]
    if not df.empty and "local_pca_tangent_cosine" in df:
        sub = df[df["method"] != "bp"]
        final = sub.sort_values("epoch").groupby(["dataset", "method", "seed", "feedback_seed"]).tail(1)
        ordered = final.groupby("method")["local_pca_tangent_cosine"].mean().sort_values()
        ax.barh([pretty_method(m) for m in ordered.index], ordered.values, color=[method_color(m) for m in ordered.index])
    ax.set_xlabel("Local-PCA tangent cosine")
    ax.set_title("Data-estimated tangent alignment")

    ax = axes[2]
    if not df.empty:
        final = df.sort_values("epoch").groupby(["dataset", "method", "seed", "feedback_seed"]).tail(1)
        grouped = final.groupby(["method", "dataset"])["test_acc"].mean().reset_index()
        methods = sorted(grouped["method"].unique(), key=method_sort_key)
        datasets = sorted(grouped["dataset"].unique())
        x = np.arange(len(methods))
        width = min(0.32, 0.75 / max(len(datasets), 1))
        for idx, dataset in enumerate(datasets):
            values = [
                grouped[(grouped["method"] == method) & (grouped["dataset"] == dataset)]["test_acc"].mean()
                for method in methods
            ]
            offset = (idx - (len(datasets) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, label=dataset)
        ax.set_xticks(x, [pretty_method(m) for m in methods], rotation=35, ha="right")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Dataset comparison")
    legend_if_handles(ax, fontsize=6.5)

    fig.suptitle("Info-DFA: standard vision-task bridge", fontsize=11.2)
    add_panel_labels(axes)
    return fig


def infodfa_all_results_dashboard() -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.0), constrained_layout=True)
    axes = axes.ravel()
    datasets = [
        ("Final accuracy by method", "dfa_synthetic/dfa_run_summary.csv"),
        ("Synthetic predictors", "dfa_synthetic/dfa_predictor_scores.csv"),
        ("Fisher over epoch", "dfa_synthetic/dfa_synthetic_results.csv"),
        ("Vision accuracy", "dfa_vision/dfa_vision_results.csv"),
        ("Vision tangent", "dfa_vision/dfa_vision_results.csv"),
        ("Vision Rayleigh", "dfa_vision/dfa_vision_results.csv"),
    ]
    for ax, (title, rel) in zip(axes, datasets):
        if rel.startswith("dfa_synthetic/"):
            df = read_first_csv(rel.replace("dfa_synthetic/", "dfa_synthetic_multimanifold/"), rel)
        elif rel.startswith("dfa_vision/"):
            df = read_vision_results()
        else:
            df = read_csv(rel)
        if "predictors" in title.lower():
            plot_predictor_bars(ax, df, color=COLORS["dfa"], label_map=LABELS)
        elif title == "Final accuracy by method" and not df.empty:
            ordered = df.groupby("method")["final_test_acc"].mean().sort_values()
            ax.barh([pretty_method(m) for m in ordered.index], ordered.values, color=[method_color(m) for m in ordered.index])
            ax.set_xlabel("Final test accuracy")
        elif title == "Fisher over epoch" and not df.empty:
            layer1 = df[df["layer"] == 1]
            for method in sorted(layer1["method"].unique(), key=method_sort_key):
                group = layer1[layer1["method"] == method]
                curve = group.groupby("epoch")["fisher_trace"].mean()
                ax.plot(curve.index, curve.values, marker="o", linewidth=1.2, label=pretty_method(method), color=method_color(method))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Whitened Fisher")
        elif "Vision" in title and not df.empty:
            metric = "test_acc" if title == "Vision accuracy" else "local_pca_tangent_cosine" if title == "Vision tangent" else "weight_rayleigh"
            final = df.sort_values("epoch").groupby(["dataset", "method", "seed", "feedback_seed"]).tail(1)
            ordered = final.groupby("method")[metric].mean().sort_values()
            ax.barh([pretty_method(m) for m in ordered.index], ordered.values, color=[method_color(m) for m in ordered.index])
            ax.set_xlabel(metric.replace("_", " "))
        else:
            add_no_data(ax, title)
        ax.set_title(title)
    fig.suptitle("Info-DFA supplementary dashboard: current result coverage", fontsize=11.5)
    add_panel_labels(axes)
    return fig


def figure_index_panel() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.1, 4.0), constrained_layout=True)
    ax.axis("off")
    ax.text(0.02, 0.92, "Info-Geo Figure Package", fontsize=16, weight="bold", transform=ax.transAxes)
    blocks = [
        ("Info-Man", ["Fig. 1 schematic", "Fig. 2 core synthetic/estimator results", "Fig. 3 communication + mixed selectivity", "Fig. 4 real-data suite", "Fig. 5 communication robustness", "Fig. 6 OpenAlyx validation", "Supplementary dashboard"]),
        ("Info-DFA", ["Fig. 1 schematic", "Fig. 2 synthetic manifold results", "Fig. 3 vision bridge", "Supplementary dashboard"]),
    ]
    y = 0.75
    for title, lines in blocks:
        ax.text(0.04, y, title, fontsize=12, weight="bold", transform=ax.transAxes)
        for idx, line in enumerate(lines):
            ax.text(0.08, y - 0.065 * (idx + 1), f"- {line}", fontsize=9.5, transform=ax.transAxes)
        y -= 0.42
    return fig


def save_figure(fig: plt.Figure, name: str, out_dir: Path, mirror_dir: Path, title: str, paper: str) -> FigureRecord:
    out_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    shared = SHARED_DRAFT
    shared.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "svg", "png"):
        path = out_dir / f"{name}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=360, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        paths.append(path)
        shutil.copy2(path, mirror_dir / path.name)
        shutil.copy2(path, shared / path.name)
    plt.close(fig)
    return FigureRecord(path=paths[0], title=title, paper=paper)


def write_manifest(records: list[FigureRecord]) -> None:
    rows = []
    for record in records:
        rows.append({"paper": record.paper, "title": record.title, "pdf": str(record.path.relative_to(ROOT))})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "figure_manifest.csv", index=False)
    lines = ["# Paper Figure Manifest", ""]
    for _, row in df.iterrows():
        lines.append(f"- **{row['paper']}**: {row['title']} -> `{row['pdf']}`")
    (OUT / "figure_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(rel: str) -> pd.DataFrame:
    path = RESULTS / rel
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_first_csv(*rels: str) -> pd.DataFrame:
    for rel in rels:
        df = read_csv(rel)
        if not df.empty:
            return df
    return pd.DataFrame()


def read_scoreboard_results() -> pd.DataFrame:
    frames = []
    root = RESULTS / "infoman_estimator_scoreboard_slurm_v4"
    if root.exists():
        for path in sorted(root.glob("*/infoman_estimator_scoreboard.csv")):
            frame = pd.read_csv(path)
            if "scenario" not in frame.columns:
                frame["scenario"] = path.parent.name
            frames.append(frame)
    if not frames:
        frame = read_csv("infoman_estimator_scoreboard/infoman_estimator_scoreboard.csv")
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def read_vision_results() -> pd.DataFrame:
    frames = []
    root = RESULTS / "dfa_vision_full_slurm_v4"
    if root.exists():
        for path in sorted(root.glob("*/dfa_vision_results.csv")):
            frame = pd.read_csv(path)
            if "dataset" not in frame.columns:
                frame["dataset"] = path.parent.name
            frames.append(frame)
    frames.extend(
        [
            read_csv("dfa_vision/dfa_vision_results.csv"),
            read_csv("dfa_vision_fashion_mnist/dfa_vision_results.csv"),
        ]
    )
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_panel_labels(axes) -> None:
    for idx, ax in enumerate(axes):
        ax.text(-0.12, 1.08, chr(ord("A") + idx), transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def legend_if_handles(ax, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, **kwargs)


def draw_node(ax, xy, text, color) -> None:
    box = FancyBboxPatch(
        (xy[0] - 0.105, xy[1] - 0.085),
        0.21,
        0.17,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.0,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.12),
    )
    ax.add_patch(box)
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=8)


def arrow_between(ax, start, stop, color="black") -> None:
    ax.add_patch(FancyArrowPatch(start, stop, arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color=color))


def draw_axes(ax) -> None:
    ax.arrow(0, 0, 2.0, 0, head_width=0.05, head_length=0.08, color="0.25", length_includes_head=True)
    ax.arrow(0, 0, 0, 1.6, head_width=0.05, head_length=0.08, color="0.25", length_includes_head=True)


def ellipse_arrow(ax, start, stop, color, label) -> None:
    ax.add_patch(FancyArrowPatch(start, stop, arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=color))
    ax.text((start[0] + stop[0]) / 2, (start[1] + stop[1]) / 2 + 0.18, label, color=color, ha="center")


def draw_network(ax, mode: str) -> None:
    layers = [[(0.1, y) for y in (0.25, 0.5, 0.75)], [(0.43, y) for y in (0.2, 0.4, 0.6, 0.8)], [(0.78, y) for y in (0.38, 0.62)]]
    for layer in layers:
        for x, y in layer:
            ax.add_patch(Circle((x, y), 0.035, facecolor="white", edgecolor="black", linewidth=1))
    for left, right in zip(layers[:-1], layers[1:]):
        for a in left:
            for b in right:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=COLORS["bp"], linewidth=0.6, alpha=0.45)
    for hidden in layers[1]:
        arrow_between(ax, (0.78, 0.5), (hidden[0] + 0.02, hidden[1]), color=COLORS["dfa"])
    ax.text(0.1, 0.93, "input", ha="center")
    ax.text(0.43, 0.93, "hidden", ha="center")
    ax.text(0.78, 0.93, "error", ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def grouped_bars(ax, df: pd.DataFrame, x_col: str, metrics: list[str], labels: list[str], colors: list[str]) -> None:
    x = np.arange(df.shape[0])
    width = 0.8 / len(metrics)
    for idx, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        if metric not in df:
            continue
        ax.bar(x + (idx - (len(metrics) - 1) / 2) * width, df[metric], width=width, label=label, color=color)
    ax.set_xticks(x, df[x_col])
    legend_if_handles(ax, fontsize=7)


def plot_scoreboard_heatmap(ax, summary: pd.DataFrame) -> None:
    if summary.empty:
        add_no_data(ax, "scoreboard")
        return
    pivot = summary.pivot(index="estimator", columns="scenario", values="abs_error_behavior")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    image = ax.imshow(pivot.values, aspect="auto", cmap="mako_r" if "mako_r" in plt.colormaps() else "viridis_r")
    ax.set_yticks(np.arange(pivot.shape[0]), [LABELS.get(x, x).replace(" ", "\n") for x in pivot.index], fontsize=6.5)
    ax.set_xticks(np.arange(pivot.shape[1]), [x.replace("_", "\n") for x in pivot.columns], fontsize=6.5)
    cbar = plt.colorbar(image, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("MAE vs behavior MI")


def plot_predictor_bars(ax, scores: pd.DataFrame, *, color: str, label_map: dict[str, str]) -> None:
    if scores.empty or "r2" not in scores:
        add_no_data(ax, "predictors")
        return
    sub = scores.replace([np.inf, -np.inf], np.nan).dropna(subset=["r2"]).sort_values("r2")
    if sub.empty:
        add_no_data(ax, "predictors")
        return
    ax.barh([label_map.get(p, p) for p in sub["predictor"]], sub["r2"], color=color)
    ax.set_xlabel(r"$R^2$")
    ax.set_xlim(0, max(1.0, float(sub["r2"].max()) * 1.05))


def plot_generic_result(ax, title: str, df: pd.DataFrame) -> None:
    if df.empty:
        add_no_data(ax, title)
        return
    if {"predictor", "r2"}.issubset(df.columns):
        plot_predictor_bars(ax, df, color=COLORS["total"], label_map=LABELS)
    elif {"estimator", "abs_error_behavior"}.issubset(df.columns):
        sub = df.groupby("estimator")["abs_error_behavior"].mean().sort_values(ascending=True).head(8)
        ax.barh([LABELS.get(x, x) for x in sub.index], sub.values, color=COLORS["total"])
        ax.set_xlabel("MAE")
    elif {"area", "used_total_ratio_mean"}.issubset(df.columns):
        sub = df.sort_values("used_total_ratio_mean").tail(8)
        ax.barh(sub["area"], sub["used_total_ratio_mean"], color=COLORS["used"])
        ax.set_xlabel("Used / total")
    elif {"variant", "used_mi_retained_fraction"}.issubset(df.columns):
        sub = df[df["variant"] != "raw"].copy()
        if "raw_used_mi" in sub:
            sub = sub[sub["raw_used_mi"] > 0.02]
        sub = sub.sort_values("used_mi_retained_fraction").tail(8)
        ax.barh(sub["area"], sub["used_mi_retained_fraction"].clip(upper=2.0), color=COLORS["bad"])
        ax.set_xlabel("Retained fraction (clip 2)")
    elif {"sender_area", "receiver_area", "triadic_transferred_used_mi"}.issubset(df.columns):
        metric = "triadic_calibrated_mi" if "triadic_calibrated_mi" in df.columns else "triadic_transferred_used_mi"
        sub = df.sort_values(metric, ascending=False).head(8).sort_values(metric)
        ax.barh([f"{s}->{r}" for s, r in zip(sub["sender_area"], sub["receiver_area"])], sub[metric], color=COLORS["comm"])
        ax.set_xlabel("Calibrated MI" if metric == "triadic_calibrated_mi" else "Triadic MI")
    else:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            add_no_data(ax, title)
        else:
            ax.plot(numeric.iloc[:, 0].to_numpy(), marker="o")
    ax.set_title(title)


def add_no_data(ax, label: str) -> None:
    ax.text(0.5, 0.5, f"No data\n{label}", ha="center", va="center", transform=ax.transAxes, color="0.45")
    ax.set_xticks([])
    ax.set_yticks([])


def pretty_method(method: str) -> str:
    return (
        method.replace("dfa_", "DFA ")
        .replace("ndfa_", "NDFA ")
        .replace("drtp_", "DRTP ")
        .replace("_", " ")
        .replace("bp", "BP")
    )


def pretty_estimator(estimator: str) -> str:
    labels = {
        "geom_total": "Total geom",
        "geom_used_cv_choice": "CV geom used",
        "projection_binned_cv_choice": "CV binned projection",
        "pica_icap_cv_choice_projection": r"Pica $I_\cap$ CV projection",
        "pica_icap_oracle_choice_projection": r"Pica $I_\cap$ oracle",
        "pica_icap_full_response_pca1": r"Pica $I_\cap$ PCA1",
        "pica_icap_full_response_pca2": r"Pica $I_\cap$ PCA2",
        "pica_icap_full_response_pca3": r"Pica $I_\cap$ PCA3",
        "neural_infonce_decoder": "InfoNCE",
        "neural_mine_decoder": "MINE",
        "neural_nwj_decoder": "NWJ",
    }
    return labels.get(estimator, estimator.replace("_", " "))


def method_sort_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else 999


def method_color(method: str) -> str:
    if method == "bp":
        return COLORS["bp"]
    if method.startswith("ndfa"):
        return COLORS["ndfa"]
    if "negative" in method or "orthogonal" in method:
        return COLORS["bad"]
    if method.startswith("drtp"):
        return COLORS["comm"]
    return COLORS["dfa"]


if __name__ == "__main__":
    main()

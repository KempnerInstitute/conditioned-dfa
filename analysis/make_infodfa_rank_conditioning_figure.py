"""Make feedback-rank and conditioning figures for the Conditioned DFA draft."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "infodfa_rank_conditioning_v1"
DRAFT_FIGS = ROOT / "drafts" / "Info-DFA" / "figures"

COLORS = {
    "bp": "#4C78A8",
    "dfa_random": "#59A14F",
    "dfa_tangent_biased": "#76B7B2",
    "dfa_bp_aligned_dynamic": "#F28E2B",
    "dfa_bp_orthogonal_dynamic": "#E15759",
    "ndfa_random": "#B07AA1",
    "ndfa_random_kronecker": "#9D755D",
    "drtp_random": "#9D9D9D",
}


def main() -> None:
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    DRAFT_FIGS.mkdir(parents=True, exist_ok=True)

    synth = load_synthetic_rank()
    vision_rank = load_vision_rank()
    damping = load_damping()

    synth.to_csv(OUT / "infodfa_synthetic_rank_summary.csv", index=False)
    vision_rank.to_csv(OUT / "infodfa_vision_rank_summary.csv", index=False)
    damping.to_csv(OUT / "infodfa_kronecker_damping_summary.csv", index=False)
    write_report(synth, vision_rank, damping)

    fig = make_figure(synth, vision_rank, damping)
    for ext in ["pdf", "svg", "png"]:
        path = OUT / f"infodfa_rank_conditioning.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 360
        fig.savefig(path, **kwargs)
        shutil.copy2(path, DRAFT_FIGS / path.name)
    plt.close(fig)
    print(f"Wrote {OUT / 'infodfa_rank_conditioning.pdf'}")


def configure() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 360,
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.3,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_synthetic_rank() -> pd.DataFrame:
    path = RESULTS / "dfa_feedback_rank_synthetic_v1" / "dfa_synthetic_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    final = (
        df.sort_values("epoch")
        .groupby(["seed", "feedback_seed", "feedback_scale", "feedback_rank", "hidden_dim", "n_hidden_layers", "task_frequency", "input_noise", "manifold", "method", "layer"], as_index=False)
        .tail(1)
    )
    run = (
        final.groupby(["seed", "feedback_seed", "feedback_scale", "feedback_rank", "hidden_dim", "n_hidden_layers", "task_frequency", "input_noise", "manifold", "method"], as_index=False)
        .agg(
            test_acc=("test_acc", "mean"),
            train_acc=("train_acc", "mean"),
            task_tangent_cosine=("task_tangent_cosine", "mean"),
            tangent_cosine=("tangent_cosine", "mean"),
            fisher_trace=("fisher_trace", "mean"),
        )
    )
    return (
        run.groupby(["manifold", "method", "feedback_rank"], as_index=False)
        .agg(
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", "sem"),
            n=("test_acc", "size"),
            task_tangent=("task_tangent_cosine", "mean"),
            tangent=("tangent_cosine", "mean"),
            fisher=("fisher_trace", "mean"),
        )
    )


def load_vision_rank() -> pd.DataFrame:
    frames = []
    for path in sorted((RESULTS / "dfa_feedback_rank_vision_v1").glob("*/*/dfa_vision_results.csv")):
        df = pd.read_csv(path)
        df["path_rank"] = float(path.parent.name.replace("rank_", ""))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    final = df.sort_values("epoch").groupby(["dataset", "method", "seed", "feedback_seed", "feedback_rank", "path_rank"], as_index=False).tail(1)
    return (
        final.groupby(["dataset", "method", "feedback_rank"], as_index=False)
        .agg(
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", "sem"),
            n=("test_acc", "size"),
            local_tangent=("local_pca_tangent_cosine", "mean"),
            rayleigh=("weight_rayleigh", "mean"),
        )
    )


def load_damping() -> pd.DataFrame:
    frames = []
    for path in sorted((RESULTS / "dfa_kronecker_damping_v1").glob("*/*/dfa_vision_results.csv")):
        damping = float(path.parent.name.replace("damping_", "").replace("p", "."))
        df = pd.read_csv(path)
        df["damping"] = damping
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    final = df.sort_values("epoch").groupby(["dataset", "method", "seed", "feedback_seed", "damping"], as_index=False).tail(1)
    return (
        final.groupby(["dataset", "method", "damping"], as_index=False)
        .agg(
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", "sem"),
            n=("test_acc", "size"),
            local_tangent=("local_pca_tangent_cosine", "mean"),
            rayleigh=("weight_rayleigh", "mean"),
        )
    )


def make_figure(synth: pd.DataFrame, vision: pd.DataFrame, damping: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.06, wspace=0.04, hspace=0.08)
    axes = axes.ravel()

    ax = axes[0]
    if not synth.empty:
        keep = synth[synth["method"].isin(["bp", "dfa_random", "dfa_tangent_biased", "drtp_random"])]
        manifold_order = ["low_rank", "swiss_roll", "torus"]
        manifold_labels = [pretty_manifold(name) for name in manifold_order]
        for method in ["bp", "dfa_random", "dfa_tangent_biased", "drtp_random"]:
            sub = keep[keep["method"] == method]
            vals = sub.groupby("manifold")["test_mean"].mean().reindex(manifold_order)
            ax.plot(manifold_labels, vals.values, marker="o", linewidth=1.5, color=COLORS.get(method, "#333333"), label=pretty(method))
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Controlled manifolds")
    ax.legend(frameon=False, fontsize=6.2)

    ax = axes[1]
    if not synth.empty:
        run = synth[synth["method"].isin(["dfa_random", "dfa_tangent_biased", "drtp_random"])].copy()
        ax.scatter(run["task_tangent"], run["test_mean"], s=26, c=[COLORS.get(m, "#333333") for m in run["method"]], alpha=0.72)
        rho = run["task_tangent"].corr(run["test_mean"], method="spearman")
        ax.text(0.03, 0.95, f"Spearman rho={rho:.2f}", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel("Task-tangent cosine")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Geometry predicts part of DFA success")

    ax = axes[2]
    if not synth.empty:
        run = synth[synth["method"].isin(["dfa_random", "dfa_tangent_biased", "drtp_random"])].copy()
        ax.scatter(np.log10(run["fisher"].clip(lower=1e-6)), run["test_mean"], s=26, c=[COLORS.get(m, "#333333") for m in run["method"]], alpha=0.72)
        rho = np.log10(run["fisher"].clip(lower=1e-6)).corr(run["test_mean"], method="spearman")
        ax.text(0.03, 0.95, f"Spearman rho={rho:.2f}", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel("log10 whitened Fisher")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Fisher is useful but incomplete")

    for ax, dataset in zip(axes[3:5], ["mnist", "fashion_mnist"]):
        sub = vision[vision["dataset"] == dataset]
        for method in ["bp", "dfa_random", "ndfa_random", "drtp_random", "dfa_bp_aligned_dynamic"]:
            m = sub[sub["method"] == method].sort_values("feedback_rank")
            if m.empty:
                continue
            ax.errorbar(m["feedback_rank"], m["test_mean"], yerr=m["test_sem"], marker="o", linewidth=1.35, color=COLORS.get(method, "#333333"), label=pretty(method))
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0, 1, 2, 4, 8, 16])
        ax.set_xticklabels(["0", "1", "2", "4", "8", "16"])
        ax.set_ylim(0.05, 1.0)
        ax.set_xlabel("Feedback rank")
        ax.set_ylabel("Final test accuracy")
        ax.set_title(dataset.replace("_", "-").upper())
    axes[3].legend(frameon=False, fontsize=5.6)

    ax = axes[5]
    if not damping.empty:
        for dataset, linestyle in [("mnist", "-"), ("fashion_mnist", "--")]:
            sub = damping[(damping["dataset"] == dataset) & (damping["method"] == "ndfa_random_kronecker")].sort_values("damping")
            ax.errorbar(sub["damping"], sub["test_mean"], yerr=sub["test_sem"], marker="o", linestyle=linestyle, linewidth=1.45, color=COLORS["ndfa_random_kronecker"], label=f"Kronecker {dataset.replace('_', '-')}")
            bp = damping[(damping["dataset"] == dataset) & (damping["method"] == "bp")]
            if not bp.empty:
                ax.axhline(bp["test_mean"].mean(), color="#4C78A8" if dataset == "mnist" else "#4C78A8", linestyle=":" if dataset == "mnist" else "-.", linewidth=1.0, alpha=0.8, label=f"BP {dataset.replace('_', '-')}")
    ax.set_xscale("log")
    ax.set_xlabel("Kronecker damping")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Conditioning transition")
    ax.legend(frameon=False, fontsize=5.3)

    fig.suptitle("Conditioned DFA: feedback rank and conditioning determine local-learning regimes", fontsize=10.6)
    for label, ax in zip("ABCDEF", axes):
        ax.text(
            0.01,
            0.99,
            label,
            transform=ax.transAxes,
            fontsize=9.6,
            weight="bold",
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5},
            zorder=20,
        )
    return fig


def pretty_manifold(name: str) -> str:
    return {
        "low_rank": "Low-rank",
        "swiss_roll": "Swiss roll",
        "torus": "Torus",
    }.get(name, name.replace("_", " "))


def pretty(method: str) -> str:
    return {
        "bp": "BP",
        "dfa_random": "DFA",
        "dfa_tangent_biased": "tangent-biased DFA",
        "dfa_bp_aligned_dynamic": "BP-aligned DFA",
        "dfa_bp_orthogonal_dynamic": "BP-orthogonal DFA",
        "ndfa_random": "nDFA",
        "ndfa_random_kronecker": "Kronecker nDFA",
        "drtp_random": "DRTP",
    }.get(method, method.replace("_", " "))


def write_report(synth: pd.DataFrame, vision: pd.DataFrame, damping: pd.DataFrame) -> None:
    lines = ["# Conditioned DFA Rank and Conditioning Results", ""]
    if not synth.empty:
        lines.append("## Synthetic")
        for manifold in sorted(synth["manifold"].unique()):
            sub = synth[synth["manifold"] == manifold]
            vals = {m: sub[sub["method"] == m]["test_mean"].mean() for m in ["bp", "dfa_random", "dfa_tangent_biased", "drtp_random"]}
            lines.append("- " + manifold + ": " + ", ".join(f"{pretty(m)}={v:.3f}" for m, v in vals.items() if pd.notna(v)))
    if not vision.empty:
        lines.append("")
        lines.append("## Vision Feedback Rank")
        for dataset in sorted(vision["dataset"].unique()):
            sub = vision[vision["dataset"] == dataset]
            for method in ["bp", "dfa_random", "ndfa_random", "drtp_random"]:
                m = sub[sub["method"] == method]
                if m.empty:
                    continue
                best = m.sort_values("test_mean", ascending=False).iloc[0]
                lines.append(f"- {dataset} {pretty(method)} best: rank={best['feedback_rank']:.0f}, acc={best['test_mean']:.3f}.")
    if not damping.empty:
        lines.append("")
        lines.append("## Kronecker Damping")
        for dataset in sorted(damping["dataset"].unique()):
            sub = damping[(damping["dataset"] == dataset) & (damping["method"] == "ndfa_random_kronecker")]
            if sub.empty:
                continue
            best = sub.sort_values("test_mean", ascending=False).iloc[0]
            lines.append(f"- {dataset}: best Kronecker damping={best['damping']:.3g}, acc={best['test_mean']:.3f}.")
    (OUT / "infodfa_rank_conditioning_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

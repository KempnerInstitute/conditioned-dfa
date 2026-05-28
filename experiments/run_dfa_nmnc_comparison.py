"""Compare DFA variants with vanilla and neural-manifold noise correlation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dfa_synthetic import (  # noqa: E402
    feedback_mode_from_method,
    method_has_dynamic_feedback,
    natural_mode_from_method,
    natural_precondition_gradients,
)
from experiments.run_dfa_vision_baselines import load_vision_dataset, minibatches  # noqa: E402
from infogeo.analysis import dataframe_to_markdown  # noqa: E402
from infogeo.dfa import (  # noqa: E402
    ManualMLP,
    gradient_cosines,
    init_fa_feedback,
    init_feedback,
    local_pca_tangent_spaces,
    tangent_projected_cosines,
)
from infogeo.noise_correlation import (  # noqa: E402
    NoiseCorrelationStats,
    activity_pca_bases,
    noise_correlation_diagnostics,
    noise_correlation_feedback_update,
    zero_feedback,
)


NOISE_CORR_METHODS = {"vnc", "nmnc"}


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_train = 1024
        args.n_test = 512
        args.epochs = 2
        args.hidden_dims = [128]
        args.eval_size = 256
        args.n_seeds = 1
        args.n_feedback_seeds = 1
        args.methods = ["bp", "dfa_random", "vnc", "nmnc"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, test_x, test_y = load_vision_dataset(args)
    train_y = corrupt_labels(train_y, n_classes=args.n_classes, noise=args.label_noise, seed=args.seed + 101)
    test_y = corrupt_labels(test_y, n_classes=args.n_classes, noise=args.test_label_noise, seed=args.seed + 202)
    rows = []
    for seed in range(args.n_seeds):
        for method in args.methods:
            feedback_seeds = [0] if method == "bp" or "bp_aligned" in method else range(args.n_feedback_seeds)
            for feedback_seed in feedback_seeds:
                rows.extend(
                    run_one(
                        train_x,
                        train_y,
                        test_x,
                        test_y,
                        method=method,
                        seed=seed,
                        feedback_seed=int(feedback_seed),
                        args=args,
                    )
                )
    df = pd.DataFrame(rows)
    csv_path = output_dir / "dfa_nmnc_results.csv"
    df.to_csv(csv_path, index=False)
    write_report(df, output_dir)
    plot_results(df, output_dir)
    print(final_summary(df).to_string(index=False))
    print(f"\nSaved results to {csv_path}")


def run_one(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    method: str,
    seed: int,
    feedback_seed: int,
    args: argparse.Namespace,
) -> list[dict[str, float | str]]:
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model = ManualMLP(
        input_dim=train_x.shape[1],
        hidden_dims=args.hidden_dims,
        output_dim=args.n_classes,
        seed=10_000 + seed,
        device=device,
    )
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    feedback = initialize_feedback(model, method=method, seed=seed, feedback_seed=feedback_seed, args=args)
    rng = np.random.default_rng(30_000 + 100 * seed + feedback_seed)
    rows = []
    eval_n = min(args.eval_size, train_x.shape[0])
    basis_n = min(args.pca_size, train_x.shape[0])
    bases = activity_pca_bases(model, train_x[:basis_n], rank=args.nc_manifold_rank)
    last_stats = empty_stats()
    step = 0
    for epoch in range(args.epochs + 1):
        bases = activity_pca_bases(model, train_x[:basis_n], rank=args.nc_manifold_rank)
        rows.append(
            evaluate(
                model,
                train_x[:eval_n],
                train_y[:eval_n],
                test_x,
                test_y,
                method,
                feedback,
                bases,
                last_stats,
                seed,
                feedback_seed,
                epoch,
                args,
            )
        )
        if epoch == args.epochs:
            break
        if feedback is not None and method_has_dynamic_feedback(method):
            feedback = initialize_feedback(model, method=method, seed=seed, feedback_seed=feedback_seed, args=args)
        for batch in minibatches(train_x.shape[0], args.batch_size, rng):
            xb = train_x[batch]
            yb = train_y[batch]
            if method in NOISE_CORR_METHODS and feedback is not None:
                if step % args.nc_pca_update_interval == 0:
                    bases = activity_pca_bases(model, train_x[:basis_n], rank=args.nc_manifold_rank)
                if step % args.nc_update_interval == 0:
                    feedback, last_stats = noise_correlation_feedback_update(
                        model,
                        xb,
                        feedback,
                        mode=method,
                        bases=bases,
                        eta=args.nc_feedback_lr,
                        noise_scale=args.nc_noise_scale,
                        manifold_rank=args.nc_manifold_rank,
                        rng=rng,
                        antithetic=not args.nc_no_antithetic,
                        normalize_by_variance=not args.nc_covariance_scaled,
                    )
            gradients = compute_gradients(model, xb, yb, method=method, feedback=feedback, args=args)
            model.apply_gradients(gradients, lr=args.lr)
            step += 1
    return rows


def initialize_feedback(
    model: ManualMLP,
    *,
    method: str,
    seed: int,
    feedback_seed: int,
    args: argparse.Namespace,
) -> list[torch.Tensor] | None:
    if method == "bp":
        return None
    if method in NOISE_CORR_METHODS and args.nc_init == "zero":
        return zero_feedback(model)
    if method.startswith("fa_"):
        return init_fa_feedback(
            model,
            seed=20_000 + 100 * seed + feedback_seed,
            scale=args.feedback_scale,
            rank=None if args.feedback_rank <= 0 else args.feedback_rank,
        )
    return init_feedback(
        model,
        mode="random" if method in NOISE_CORR_METHODS else feedback_mode_from_method(method),
        seed=20_000 + 100 * seed + feedback_seed,
        scale=args.feedback_scale,
        rank=None if args.feedback_rank <= 0 else args.feedback_rank,
    )


def compute_gradients(
    model: ManualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    method: str,
    feedback,
    args: argparse.Namespace,
):
    if method == "bp":
        return model.bp_gradients(x, y)
    if feedback is None:
        raise ValueError(f"{method} requires feedback")
    if method.startswith("drtp_"):
        gradients = model.target_projection_gradients(x, y, feedback)
    elif method.startswith("fa_"):
        gradients = model.fa_gradients(x, y, feedback)
    else:
        gradients = model.dfa_gradients(x, y, feedback)
    if method.startswith("ndfa_"):
        gradients = natural_precondition_gradients(
            model,
            gradients,
            x,
            damping=args.natural_damping,
            mode=natural_mode_from_method(method),
        )
    return gradients


def evaluate(
    model: ManualMLP,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    method: str,
    feedback,
    bases,
    stats: NoiseCorrelationStats,
    seed: int,
    feedback_seed: int,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    bp = model.bp_gradients(x_eval, y_eval)
    diagnostics = {
        "param_cosine": np.nan,
        "activity_cosine_mean": np.nan,
        "activity_angle_deg_mean": np.nan,
        "projected_step_ratio_mean": np.nan,
        "activity_norm_ratio_mean": np.nan,
        "manifold_gradient_alpha_mean": np.nan,
        "manifold_dim_fraction_mean": np.nan,
        "manifold_condition_margin_mean": np.nan,
        "local_pca_tangent_cosine": np.nan,
    }
    if feedback is not None:
        local = compute_gradients(model, x_eval, y_eval, method=method, feedback=feedback, args=args)
        alignments = gradient_cosines(bp, local)
        diagnostics.update({key: float(value) for key, value in alignments.items() if np.isscalar(value)})
        diagnostics.update(noise_correlation_diagnostics(bp, local, bases))
        hidden = [h.detach().cpu().numpy() for h in model.hidden_activations(x_eval)]
        tangents = [
            local_pca_tangent_spaces(h, n_neighbors=args.local_pca_neighbors, rank=args.local_pca_rank)
            for h in hidden
        ]
        tangent_alignments = tangent_projected_cosines(bp, local, tangents)
        finite_tangent = [value for value in tangent_alignments.values() if np.isfinite(value)]
        diagnostics["local_pca_tangent_cosine"] = float(np.mean(finite_tangent)) if finite_tangent else np.nan
        if args.covariance_diagnostics:
            diagnostics.update(covariance_diagnostics(model, x_eval, bp, local, damping=args.natural_damping))
    elif args.covariance_diagnostics:
        diagnostics.update(covariance_diagnostics(model, x_eval, bp, bp, damping=args.natural_damping))
    return {
        "dataset": args.dataset,
        "method": method,
        "seed": float(seed),
        "feedback_seed": float(feedback_seed),
        "feedback_rank": float(args.feedback_rank),
        "nc_update_interval": float(args.nc_update_interval),
        "nc_manifold_rank": float(args.nc_manifold_rank),
        "nc_noise_scale": float(args.nc_noise_scale),
        "nc_feedback_lr": float(args.nc_feedback_lr),
        "natural_damping": float(args.natural_damping),
        "n_train": float(args.n_train),
        "n_test": float(args.n_test),
        "label_noise": float(args.label_noise),
        "test_label_noise": float(args.test_label_noise),
        "epoch": float(epoch),
        "loss": bp.loss,
        "train_eval_acc": model.accuracy(x_eval, y_eval),
        "test_acc": model.accuracy(x_test, y_test),
        "feedback_norm": stats.mean_feedback_norm,
        "feedback_update_norm": stats.mean_update_norm,
        "noise_norm": stats.mean_noise_norm,
        "delta_output_norm": stats.mean_delta_output_norm,
        **diagnostics,
    }


def corrupt_labels(y: torch.Tensor, *, n_classes: int, noise: float, seed: int) -> torch.Tensor:
    """Randomly replace a fraction of labels with an incorrect class."""

    noise = float(noise)
    if noise <= 0:
        return y
    generator = torch.Generator(device=y.device)
    generator.manual_seed(int(seed))
    mask = torch.rand(y.shape, generator=generator, device=y.device) < min(noise, 1.0)
    if not bool(mask.any()):
        return y
    offsets = torch.randint(1, int(n_classes), (int(mask.sum().item()),), generator=generator, device=y.device)
    out = y.clone()
    out[mask] = (out[mask] + offsets) % int(n_classes)
    return out


def covariance_diagnostics(
    model: ManualMLP,
    x: torch.Tensor,
    bp: object,
    local: object,
    *,
    damping: float,
) -> dict[str, float]:
    with torch.no_grad():
        _, activations, _ = model.forward(x)
    presynaptic_stats = []
    local_delta_stats = []
    bp_delta_stats = []
    for layer_idx in range(model.n_hidden_layers):
        presynaptic_stats.append(spectrum_stats(activations[layer_idx], damping=damping))
        local_delta_stats.append(spectrum_stats(local.deltas[layer_idx], damping=damping))
        bp_delta_stats.append(spectrum_stats(bp.deltas[layer_idx], damping=damping))
    return {
        **prefix_mean_stats("pre_activity", presynaptic_stats),
        **prefix_mean_stats("local_error", local_delta_stats),
        **prefix_mean_stats("bp_error", bp_delta_stats),
    }


def prefix_mean_stats(prefix: str, stats: list[dict[str, float]]) -> dict[str, float]:
    if not stats:
        return {}
    keys = sorted(stats[0])
    return {f"{prefix}_{key}_mean": float(np.nanmean([item[key] for item in stats])) for key in keys}


def spectrum_stats(values: torch.Tensor, *, damping: float, eps: float = 1e-12) -> dict[str, float]:
    x = values.detach().flatten(start_dim=1).float()
    if x.shape[0] <= 1 or x.shape[1] == 0:
        return {"condition": np.nan, "effective_rank": np.nan, "top1_fraction": np.nan, "trace": np.nan}
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(dim=0, keepdim=True)
    try:
        singular = torch.linalg.svdvals(x.cpu())
    except RuntimeError:
        cov = (x.T @ x) / max(int(x.shape[0]) - 1, 1)
        cov = 0.5 * (cov + cov.T)
        eig = robust_eigvalsh(cov)
    else:
        eig = singular.square() / max(int(x.shape[0]) - 1, 1)
    eig = torch.sort(torch.nan_to_num(eig, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)).values
    trace = eig.sum().clamp_min(eps)
    probs = eig / trace
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    effective_rank = torch.exp(entropy)
    top1 = eig[-1] / trace
    damp = max(float(damping), eps)
    condition = (eig[-1] + damp) / (eig[0] + damp)
    return {
        "condition": float(condition.detach().cpu().item()),
        "effective_rank": float(effective_rank.detach().cpu().item()),
        "top1_fraction": float(top1.detach().cpu().item()),
        "trace": float(trace.detach().cpu().item()),
    }


def robust_eigvalsh(cov: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    for jitter in (0.0, 1e-8, 1e-6, 1e-4):
        try:
            return torch.linalg.eigvalsh(cov + jitter * eye)
        except RuntimeError:
            continue
    diag = torch.diagonal(cov).clamp_min(0.0)
    return diag


def empty_stats() -> NoiseCorrelationStats:
    return NoiseCorrelationStats(
        mean_feedback_norm=float("nan"),
        mean_update_norm=float("nan"),
        mean_noise_norm=float("nan"),
        mean_delta_output_norm=float("nan"),
    )


def final_summary(df: pd.DataFrame) -> pd.DataFrame:
    final = df.sort_values("epoch").groupby(["dataset", "method", "seed", "feedback_seed"]).tail(1)
    return (
        final.groupby(["dataset", "method"], as_index=False)
        .agg(
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", "sem"),
            activity_angle=("activity_angle_deg_mean", "mean"),
            projected_step=("projected_step_ratio_mean", "mean"),
            alpha=("manifold_gradient_alpha_mean", "mean"),
            margin=("manifold_condition_margin_mean", "mean"),
            n=("test_acc", "size"),
        )
        .sort_values(["dataset", "test_mean"], ascending=[True, False])
    )


def write_report(df: pd.DataFrame, output_dir: Path) -> None:
    summary = final_summary(df)
    lines = [
        "# DFA/NMNC Comparison",
        "",
        "## Design",
        "",
        "Vanilla noise correlation (VNC) learns feedback from full-space hidden perturbations. Neural manifold noise correlation (NMNC) restricts perturbations to global PCA activity bases. Both are compared with BP, DFA, nDFA, Kronecker nDFA, and DRTP when requested.",
        "",
        "## Final Summary",
        "",
        dataframe_to_markdown(summary, float_format=".4f"),
        "",
    ]
    (output_dir / "dfa_nmnc_report.md").write_text("\n".join(lines))


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for method, group in df.groupby("method"):
        curve = group.groupby("epoch")["test_acc"].mean()
        axes[0, 0].plot(curve.index, curve.values, marker="o", label=method)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Test accuracy")
    axes[0, 0].legend(frameon=False, fontsize=7)

    final = df.sort_values("epoch").groupby(["method", "seed", "feedback_seed"]).tail(1)
    by_method = final.groupby("method")["test_acc"].mean().sort_values()
    axes[0, 1].barh(by_method.index, by_method.values)
    axes[0, 1].set_xlabel("Final test accuracy")

    for method, group in df[df["method"].isin(["vnc", "nmnc"])].groupby("method"):
        axes[1, 0].plot(group.groupby("epoch")["activity_angle_deg_mean"].mean(), marker="o", label=method)
        axes[1, 1].plot(group.groupby("epoch")["projected_step_ratio_mean"].mean(), marker="o", label=method)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Activation-gradient angle to BP (deg)")
    axes[1, 0].legend(frameon=False, fontsize=7)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Projected step ratio")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    fig.savefig(output_dir / "dfa_nmnc_summary.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="results/dfa_nmnc_comparison")
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist", "cifar10", "cifar100"])
    parser.add_argument("--data-dir", default="data/torchvision")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker", "drtp_random", "vnc", "nmnc"],
    )
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-feedback-seeds", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--label-noise", type=float, default=0.0, help="Fraction of training labels replaced by an incorrect class.")
    parser.add_argument("--test-label-noise", type=float, default=0.0, help="Optional test-label corruption; defaults to clean evaluation.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--feedback-rank", type=int, default=0)
    parser.add_argument("--natural-damping", type=float, default=0.3)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--pca-size", type=int, default=1024)
    parser.add_argument("--local-pca-neighbors", type=int, default=24)
    parser.add_argument("--local-pca-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--nc-update-interval", type=int, default=10)
    parser.add_argument("--nc-pca-update-interval", type=int, default=20)
    parser.add_argument("--nc-manifold-rank", type=int, default=32)
    parser.add_argument("--nc-noise-scale", type=float, default=0.05)
    parser.add_argument("--nc-feedback-lr", type=float, default=0.2)
    parser.add_argument("--nc-init", choices=["zero", "random"], default="zero")
    parser.add_argument("--nc-no-antithetic", action="store_true")
    parser.add_argument("--nc-covariance-scaled", action="store_true")
    parser.add_argument("--covariance-diagnostics", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

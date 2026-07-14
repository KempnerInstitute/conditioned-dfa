"""Compare DFA-Stall dynamics under raw DFA, nDFA, and K-nDFA.

This script keeps the experimental setup from the external DFA-Stall repository:
a three-hidden-layer tanh MLP trained on one-hot MNIST-family data with direct feedback.
It then applies the Info-DFA preconditioners to the same hidden gradients:

* ``dfa``: raw direct feedback alignment.
* ``ndfa``: input/activity-side second-moment preconditioning.
* ``endfa``: error/local-delta-side second-moment preconditioning.
* ``kndfa``: activity-side plus local-error-side Kronecker preconditioning.
* ``kndfa_bp``: the K-nDFA update with a nonlocal BP-error covariance source.

The external repo is intentionally kept under ``external/DFA-Stall`` and ignored
by git. Clone it before running:

    git clone git@github.com:varun04reddy/DFA-Stall.git external/DFA-Stall
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external" / "DFA-Stall"

METHOD_LABEL = {
    "bp": "BP",
    "dfa": "DFA",
    "ndfa": "nDFA",
    "endfa": "error-nDFA",
    "kndfa": "K-nDFA",
    "kndfa_bp": "K-nDFA (BP-error source)",
}
METHOD_COLOR = {
    "bp": "#222222",
    "dfa": "#0072B2",
    "ndfa": "#009E73",
    "endfa": "#D55E00",
    "kndfa": "#6A3D9A",
    "kndfa_bp": "#CC79A7",
}


def load_dfa_stall_module():
    path = EXTERNAL / "train.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Clone git@github.com:varun04reddy/DFA-Stall.git "
            "into external/DFA-Stall first."
        )
    spec = importlib.util.spec_from_file_location("dfa_stall_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/dfa_stall_comparison_v1")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("--methods", nargs="+", default=["bp", "dfa", "ndfa", "kndfa"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--feedback-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=0.3, help="Activity-side covariance damping.")
    parser.add_argument(
        "--error-damping",
        type=float,
        default=None,
        help="Error-side covariance damping for error-nDFA/K-nDFA (default: reuse --damping).",
    )
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument(
        "--norm-match-hidden",
        action="store_true",
        help="Rescale each preconditioned hidden weight gradient to the raw-DFA norm.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-n", type=int, default=1024)
    parser.add_argument(
        "--validation-size",
        type=int,
        default=0,
        help="Hold out this many training examples for damping selection; zero retains the legacy test probe.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=12_345,
        help="Seed for the fixed train/validation split.",
    )
    parser.add_argument(
        "--feature-probe-n",
        type=int,
        default=0,
        help="Examples used for feature-path tracking (zero uses the whole evaluation probe).",
    )
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--metric-every", type=int, default=1)
    parser.add_argument("--max-train", type=int, default=0, help="Optional train-set subset for quick tests.")
    parser.add_argument("--max-test", type=int, default=0, help="Optional test-set subset for quick tests.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stall = load_dfa_stall_module()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    x_train, y_train, x_test, y_test = load_dataset(stall, args.dataset, Path(args.data_dir))
    if args.validation_size > 0:
        n_val = min(int(args.validation_size), int(x_train.shape[0]) - 1)
        split_gen = torch.Generator(device="cpu").manual_seed(int(args.validation_seed))
        order = torch.randperm(int(x_train.shape[0]), generator=split_gen)
        val_idx, train_idx = order[:n_val], order[n_val:]
        x_probe, y_probe = x_train[val_idx], y_train[val_idx]
        x_train, y_train = x_train[train_idx], y_train[train_idx]
        probe_split = "validation"
    else:
        x_probe, y_probe = x_test, y_test
        probe_split = "test"
    if args.max_train > 0:
        x_train, y_train = x_train[: args.max_train], y_train[: args.max_train]
    if args.max_test > 0:
        x_test, y_test = x_test[: args.max_test], y_test[: args.max_test]

    rows = []
    for seed in args.seeds:
        for feedback_seed in args.feedback_seeds:
            for method in args.methods:
                rows.extend(
                    run_one(
                        stall,
                        args,
                        method,
                        seed,
                        feedback_seed,
                        device,
                        x_train,
                        y_train,
                        x_probe,
                        y_probe,
                        x_test,
                        y_test,
                        probe_split,
                    )
                )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "dfa_stall_comparison_results.csv", index=False)
    summary = summarize(df)
    summary.to_csv(output_dir / "dfa_stall_comparison_summary.csv", index=False)
    write_report(df, summary, output_dir, args)
    if not args.no_plots:
        make_plots(df, summary, output_dir)
    print(summary.to_string(index=False))
    print(f"\nwrote {output_dir}")


def load_dataset(stall, dataset: str, data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dataset == "mnist":
        return stall.load_mnist(data_dir)
    if dataset != "fashion_mnist":
        raise ValueError(f"unsupported dataset: {dataset}")
    from torchvision import datasets

    train = datasets.FashionMNIST(root=str(data_dir), train=True, download=True)
    test = datasets.FashionMNIST(root=str(data_dir), train=False, download=True)

    def flatten(data) -> torch.Tensor:
        return data.data.float().div(255.0).view(-1, 784)

    return (
        flatten(train),
        torch.as_tensor(train.targets, dtype=torch.long),
        flatten(test),
        torch.as_tensor(test.targets, dtype=torch.long),
    )


def run_one(
    stall,
    args: argparse.Namespace,
    method: str,
    seed: int,
    feedback_seed: int,
    device: torch.device,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    probe_split: str,
) -> list[dict]:
    n_classes = int(max(y_train.max().item(), y_test.max().item())) + 1
    torch.manual_seed(seed)
    model = stall.TanhMLP(int(x_train.shape[1]), args.hidden, n_classes, seed).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    feedback = init_feedback(args, n_classes, device, seed=seed, feedback_seed=feedback_seed) if method != "bp" else None

    probe_n = min(int(args.probe_n), int(x_probe.shape[0]))
    probe_x = x_probe[:probe_n].to(device).float()
    probe_t = stall.to_one_hot(y_probe[:probe_n], n_classes).to(device)
    probe_y = y_probe[:probe_n].to(device)
    feature_n = probe_n if args.feature_probe_n <= 0 else min(int(args.feature_probe_n), probe_n)
    feature_x = probe_x[:feature_n]

    test_x = x_test.to(device).float()
    test_y = y_test.to(device)
    test_t = stall.to_one_hot(test_y, n_classes).to(device)

    with torch.no_grad():
        model(feature_x)
        h_prev = [h.clone() for h in model.acts]
        h0_norm2 = [float(h.pow(2).sum(1).mean().item()) + 1e-8 for h in h_prev]
        feature_path = np.zeros(model.layers.__len__() - 1, dtype=np.float64)

    rng = torch.Generator(device="cpu").manual_seed(seed + 100)
    rows: list[dict] = []
    n = int(x_train.shape[0])

    for step in range(1, args.total_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), generator=rng)
        xb = x_train[idx].to(device).float()
        targets = stall.to_one_hot(y_train[idx], n_classes).to(device)
        record_step = step % args.metric_every == 0 or step == 1

        bp_grads = None
        bp_loss = None
        if method == "bp" or record_step:
            bp_grads, bp_loss = bp_gradients(stall, model, xb, targets)
        if method == "bp":
            if bp_grads is None or bp_loss is None:
                raise RuntimeError("BP gradients were not computed for BP method")
            method_grads = bp_grads
            teachings = []
            g_primes = []
            loss = bp_loss
        else:
            method_grads, teachings, g_primes, loss = dfa_family_gradients(
                stall,
                model,
                xb,
                targets,
                feedback,
                method=method,
                activity_damping=args.damping,
                error_damping=args.damping if args.error_damping is None else args.error_damping,
                norm_match_hidden=args.norm_match_hidden,
            )

        opt.zero_grad(set_to_none=True)
        for layer, (gw, gb) in zip(model.layers, method_grads):
            layer.weight.grad = gw.detach().clone()
            layer.bias.grad = gb.detach().clone()
        opt.step()

        if record_step:
            if bp_grads is None:
                bp_grads, _ = bp_gradients(stall, model, xb, targets)
            row = base_metrics(stall, model, method, seed, feedback_seed, step, loss, method_grads, bp_grads)
            row["dataset"] = args.dataset
            row["probe_split"] = probe_split
            row["activity_damping"] = float(args.damping)
            row["error_damping"] = float(args.damping if args.error_damping is None else args.error_damping)
            row.update(layer_metrics(stall, model, method, feedback, method_grads, teachings, g_primes, feature_path))
            if step % args.eval_every == 0 or step == 1 or step == args.total_steps:
                row["probe_loss"] = loss_on(stall, model, probe_x, probe_t)
                row["probe_acc"] = accuracy(model, probe_x, probe_y)
            if step == args.total_steps:
                row["test_loss"] = loss_on(stall, model, test_x, test_t)
                row["test_acc"] = accuracy(model, test_x, test_y)
            rows.append(row)

        with torch.no_grad():
            model(feature_x)
            for li, h_curr in enumerate(model.acts):
                d2 = (h_curr - h_prev[li]).pow(2).sum(1).mean().item()
                feature_path[li] += d2 / h0_norm2[li]
            h_prev = [h.clone() for h in model.acts]

        if step % 250 == 0 or step == args.total_steps:
            print(
                f"method={method:5s} seed={seed} fb={feedback_seed} "
                f"step={step:5d}/{args.total_steps} loss={loss:.4f}"
            )

    return rows


def init_feedback(args: argparse.Namespace, output_dim: int, device: torch.device, *, seed: int, feedback_seed: int) -> list[torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(20_000 + 100 * seed + feedback_seed)
    scale = float(args.feedback_scale)
    return [scale * torch.randn(args.hidden, output_dim, generator=gen, device=device) for _ in range(3)]


def bp_gradients(stall, model, xb: torch.Tensor, targets: torch.Tensor) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], float]:
    model.zero_grad(set_to_none=True)
    preds = model(xb)
    loss = stall.binary_log_loss(targets, preds)
    loss.backward()
    grads = [(layer.weight.grad.detach().clone(), layer.bias.grad.detach().clone()) for layer in model.layers]
    return grads, float(loss.item())


@torch.no_grad()
def dfa_family_gradients(
    stall,
    model,
    xb: torch.Tensor,
    targets: torch.Tensor,
    feedback: list[torch.Tensor],
    *,
    method: str,
    activity_damping: float,
    error_damping: float,
    norm_match_hidden: bool,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[torch.Tensor], list[torch.Tensor], float]:
    preds = model(xb)
    error = preds - targets
    loss = float(stall.binary_log_loss(targets, preds).item())
    bp_hidden_deltas = exact_bp_hidden_deltas(model, error) if method == "kndfa_bp" else []
    grads: list[tuple[torch.Tensor, torch.Tensor]] = []
    teachings: list[torch.Tensor] = []
    g_primes: list[torch.Tensor] = []

    for li, layer in enumerate(model.layers[:-1]):
        h_prev_l = xb if li == 0 else model.acts[li - 1]
        direct = error @ feedback[li].t()
        gp = 1.0 - model.preacts[li].tanh().pow(2)
        delta = direct * gp
        gw = delta.t() @ h_prev_l / xb.shape[0]
        gb = delta.mean(0)
        raw_norm = gw.norm().clamp_min(1e-12)
        if method in {"ndfa", "kndfa", "kndfa_bp"}:
            cov_activity = h_prev_l.t() @ h_prev_l / max(int(h_prev_l.shape[0]), 1)
            gw = damped_solve(cov_activity, gw.t(), damping=activity_damping).t()
        if method in {"endfa", "kndfa", "kndfa_bp"}:
            error_source = bp_hidden_deltas[li] if method == "kndfa_bp" else delta
            cov_error = error_source.t() @ error_source / max(int(error_source.shape[0]), 1)
            gw = damped_solve(cov_error, gw, damping=error_damping)
        if norm_match_hidden and method in {"ndfa", "endfa", "kndfa", "kndfa_bp"}:
            gw = gw * (raw_norm / gw.norm().clamp_min(1e-12))
        grads.append((gw, gb))
        teachings.append(delta)
        g_primes.append(gp)

    h_last = model.acts[-1]
    grads.append((error.t() @ h_last / xb.shape[0], error.mean(0)))
    return grads, teachings, g_primes, loss


@torch.no_grad()
def exact_bp_hidden_deltas(model, output_error: torch.Tensor) -> list[torch.Tensor]:
    """Per-example BP hidden deltas, used only as a nonlocal covariance-source control."""

    hidden_deltas: list[torch.Tensor] = [torch.empty(0, device=output_error.device) for _ in model.layers[:-1]]
    delta_next = output_error
    for li in range(len(model.layers) - 2, -1, -1):
        delta_next = (delta_next @ model.layers[li + 1].weight) * (1.0 - model.preacts[li].tanh().pow(2))
        hidden_deltas[li] = delta_next
    return hidden_deltas


def damped_solve(cov: torch.Tensor, rhs: torch.Tensor, *, damping: float) -> torch.Tensor:
    cov = 0.5 * (cov + cov.t())
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    base = max(float(damping), 1e-6)
    last_error: RuntimeError | None = None
    for multiplier in (1.0, 10.0, 100.0, 1000.0):
        try:
            out = torch.linalg.solve(cov + (base * multiplier) * eye, rhs)
        except RuntimeError as exc:
            last_error = exc
            continue
        if torch.isfinite(out).all():
            return out
    if last_error is not None:
        raise last_error
    raise RuntimeError("damped solve failed")


def base_metrics(stall, model, method: str, seed: int, feedback_seed: int, step: int, loss: float, method_grads, bp_grads) -> dict:
    hidden = range(len(model.layers) - 1)
    method_cat = torch.cat([method_grads[li][0].reshape(-1) for li in hidden])
    bp_cat = torch.cat([bp_grads[li][0].reshape(-1) for li in hidden])
    row = {
        "method": method,
        "method_label": METHOD_LABEL.get(method, method),
        "seed": seed,
        "feedback_seed": feedback_seed,
        "step": step,
        "train_loss": float(loss),
        "grad_alignment": stall.cosine_flat(method_cat, bp_cat),
        "probe_loss": np.nan,
        "probe_acc": np.nan,
        "test_loss": np.nan,
        "test_acc": np.nan,
    }
    for li in hidden:
        row[f"param_grad_alignment_l{li + 1}"] = stall.cosine_flat(method_grads[li][0], bp_grads[li][0])
        row[f"grad_norm_l{li + 1}"] = float(method_grads[li][0].norm().item())
    return row


@torch.no_grad()
def layer_metrics(stall, model, method: str, feedback, method_grads, teachings, g_primes, feature_path: np.ndarray) -> dict:
    out: dict[str, float] = {}
    eff_maps = stall.effective_maps(model)
    for li in range(len(model.layers) - 1):
        lid = li + 1
        out[f"feature_path_l{lid}"] = float(feature_path[li])
        out[f"angular_update_l{lid}"] = stall.angular_update(model.layers[li].weight.detach(), method_grads[li][0].detach())
        if method == "bp":
            out[f"gate_participation_l{lid}"] = np.nan
            out[f"gate_strength_sq_l{lid}"] = np.nan
            out[f"saturation_frac_l{lid}"] = np.nan
            out[f"feedback_signal_norm_l{lid}"] = np.nan
            out[f"weight_alignment_l{lid}"] = np.nan
            continue
        gs = stall.gate_stats(g_primes[li])
        out[f"gate_participation_l{lid}"] = gs["gate_participation"]
        out[f"gate_strength_sq_l{lid}"] = gs["gate_strength_sq"]
        out[f"saturation_frac_l{lid}"] = gs["saturation_frac"]
        out[f"feedback_signal_norm_l{lid}"] = float(teachings[li].norm().item() / math.sqrt(float(teachings[li].numel())))
        eff = eff_maps[li]
        fb = feedback[li]
        dots = (eff * fb).sum(1)
        denom = eff.norm(1).clamp_min(1e-12) * fb.norm(1).clamp_min(1e-12)
        out[f"weight_alignment_l{lid}"] = float((dots / denom).mean().item())
    return out


@torch.no_grad()
def loss_on(stall, model, x: torch.Tensor, targets: torch.Tensor, chunk_size: int = 2048) -> float:
    losses: list[float] = []
    weights: list[int] = []
    for start in range(0, x.shape[0], chunk_size):
        xb = x[start : start + chunk_size]
        yb = targets[start : start + chunk_size]
        losses.append(float(stall.binary_log_loss(yb, model(xb)).item()))
        weights.append(int(xb.shape[0]))
    return float(np.average(losses, weights=weights))


@torch.no_grad()
def accuracy(model, x: torch.Tensor, y: torch.Tensor, chunk_size: int = 2048) -> float:
    correct = 0
    total = 0
    for start in range(0, x.shape[0], chunk_size):
        xb = x[start : start + chunk_size]
        yb = y[start : start + chunk_size]
        pred = torch.argmax(model(xb), dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(["method", "seed", "feedback_seed"], sort=False):
        method, seed, feedback_seed = keys
        sub = sub.sort_values("step")
        stall_s, stall_e = detect_stall_steps(sub["train_loss"].to_numpy(), sub["step"].to_numpy())
        eval_sub = sub.dropna(subset=["probe_acc"])
        final_eval = eval_sub.iloc[-1] if not eval_sub.empty else sub.iloc[-1]
        init_loss = float(sub["train_loss"].iloc[0])
        final_loss = float(sub["train_loss"].iloc[-1])
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABEL.get(method, method),
                "seed": seed,
                "feedback_seed": feedback_seed,
                "stall_start_step": stall_s,
                "stall_end_step": stall_e,
                "stall_duration": max(0, stall_e - stall_s),
                "init_loss": init_loss,
                "final_loss": final_loss,
                "loss_drop": init_loss - final_loss,
                "best_loss": float(sub["train_loss"].min()),
                "mean_grad_alignment": float(sub["grad_alignment"].mean()),
                "final_grad_alignment": float(sub["grad_alignment"].iloc[-1]),
                "final_probe_loss": float(final_eval.get("probe_loss", np.nan)),
                "final_probe_acc": float(final_eval.get("probe_acc", np.nan)),
                "final_test_loss": float(sub["test_loss"].dropna().iloc[-1]) if sub["test_loss"].notna().any() else np.nan,
                "final_test_acc": float(sub["test_acc"].dropna().iloc[-1]) if sub["test_acc"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def detect_stall_steps(loss: np.ndarray, steps: np.ndarray, smooth_w: int = 100) -> tuple[int, int]:
    if loss.size < 5:
        return int(steps[0]), int(steps[-1])
    step_spacing = float(np.nanmedian(np.diff(steps))) if steps.size > 1 else 1.0
    step_spacing = max(step_spacing, 1.0)
    w = min(max(5, int(round(float(smooth_w) / step_spacing))), int(loss.size))
    smoothed = np.convolve(loss, np.ones(w) / w, mode="same")
    velocity = -np.gradient(smoothed)
    skip = min(max(1, int(round(50.0 / step_spacing))), loss.size // 4)
    early_w = min(max(5, int(round(200.0 / step_spacing))), loss.size - skip)
    early = velocity[skip : skip + early_w]
    if not len(early) or np.nanmax(early) <= 0:
        return int(steps[loss.size // 4]), int(steps[3 * loss.size // 4])
    threshold = 0.05 * np.nanmax(early)
    idxs = np.where((velocity < threshold) & (np.arange(loss.size) >= skip))[0]
    if not len(idxs):
        return int(steps[loss.size // 4]), int(steps[3 * loss.size // 4])
    start = int(idxs[0])
    recovery_gap = min(max(1, int(round(50.0 / step_spacing))), loss.size // 10)
    rec = np.where((velocity > threshold) & (np.arange(loss.size) > start + recovery_gap))[0]
    end = int(rec[0]) if len(rec) else loss.size - 1
    return int(steps[start]), int(steps[end])


def write_report(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    mean = (
        summary.groupby(["method", "method_label"], as_index=False)
        .agg(
            final_probe_acc_mean=("final_probe_acc", "mean"),
            final_probe_acc_sem=("final_probe_acc", sem),
            final_test_acc_mean=("final_test_acc", "mean"),
            final_test_acc_sem=("final_test_acc", sem),
            loss_drop_mean=("loss_drop", "mean"),
            stall_duration_mean=("stall_duration", "mean"),
            mean_grad_alignment=("mean_grad_alignment", "mean"),
            n=("seed", "count"),
        )
        .sort_values("method")
    )
    lines = [
        "# DFA-Stall comparison",
        "",
        f"- External reference repo: `{EXTERNAL}`",
        f"- Dataset: {args.dataset} (no injected input or label noise).",
        f"- Steps: {args.total_steps}; hidden units: {args.hidden}; batch size: {args.batch_size}; "
        f"lr: {args.lr:g}; activity damping: {args.damping:g}; error damping: "
        f"{args.damping if args.error_damping is None else args.error_damping:g}.",
        f"- Selection probe: {'fixed training-validation split' if args.validation_size > 0 else 'legacy test subset'}; "
        f"probe examples: {min(args.probe_n, args.validation_size if args.validation_size > 0 else 10_000)}.",
        f"- Methods: {', '.join(args.methods)}.",
        "",
        "## Mean by method",
        "",
        mean.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Per-run summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Interpretation guide: the detected stall window is a heuristic on the logged loss curve.",
        "The main readouts are the loss/probe-accuracy dynamics, final probe accuracy, loss drop,",
        "and DFA-vs-BP gradient alignment.",
    ]
    (output_dir / "dfa_stall_comparison_summary.md").write_text("\n".join(lines) + "\n")


def sem(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def make_plots(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    style()
    plot_learning_curves(df, summary, output_dir)
    plot_alignment_and_gates(df, output_dir)
    plot_stall_summary(summary, output_dir)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
        }
    )


def plot_learning_curves(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), constrained_layout=True)
    for method, sub in df.groupby("method", sort=False):
        curve = sub.groupby("step", as_index=False).agg(train_loss=("train_loss", "mean"), probe_acc=("probe_acc", "mean"))
        axes[0].plot(curve["step"], smooth(curve["train_loss"]), color=METHOD_COLOR.get(method), label=METHOD_LABEL.get(method, method))
        eval_curve = curve.dropna(subset=["probe_acc"])
        axes[1].plot(eval_curve["step"], eval_curve["probe_acc"] * 100.0, marker="o", ms=2.5, color=METHOD_COLOR.get(method), label=METHOD_LABEL.get(method, method))
        s = summary[summary["method"] == method]
        if not s.empty and method != "bp":
            lo = float(s["stall_start_step"].mean())
            hi = float(s["stall_end_step"].mean())
            axes[0].axvspan(lo, hi, color=METHOD_COLOR.get(method), alpha=0.08, lw=0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("train loss")
    axes[0].set_title("Loss dynamics")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("probe accuracy (%)")
    axes[1].set_title("Held-out probe")
    axes[0].legend(frameon=False, ncol=2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"dfa_stall_learning_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_alignment_and_gates(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), constrained_layout=True)
    for method, sub in df.groupby("method", sort=False):
        curve = sub.groupby("step", as_index=False).mean(numeric_only=True)
        color = METHOD_COLOR.get(method)
        label = METHOD_LABEL.get(method, method)
        axes[0].plot(curve["step"], smooth(curve["grad_alignment"]), color=color, label=label)
        gate_cols = [c for c in curve.columns if c.startswith("gate_participation_l")]
        path_cols = [c for c in curve.columns if c.startswith("feature_path_l")]
        if gate_cols:
            axes[1].plot(curve["step"], smooth(curve[gate_cols].mean(axis=1)), color=color, label=label)
        if path_cols:
            axes[2].plot(curve["step"], smooth(curve[path_cols].mean(axis=1)), color=color, label=label)
    axes[0].set_title("Gradient alignment")
    axes[0].set_ylabel(r"$\cos(g, g_{\rm BP})$")
    axes[1].set_title("Gate participation")
    axes[1].set_ylabel("mean across layers")
    axes[2].set_title("Feature movement")
    axes[2].set_ylabel("path length")
    for ax in axes:
        ax.set_xlabel("step")
    axes[0].legend(frameon=False)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"dfa_stall_mechanism_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stall_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    mean = (
        summary.groupby(["method", "method_label"], as_index=False)
        .agg(
            stall_duration=("stall_duration", "mean"),
            loss_drop=("loss_drop", "mean"),
            final_probe_acc=("final_probe_acc", "mean"),
        )
        .sort_values("method")
    )
    x = np.arange(mean.shape[0])
    colors = [METHOD_COLOR.get(m, "#777777") for m in mean["method"]]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)
    for ax, col, title in [
        (axes[0], "stall_duration", "Detected stall"),
        (axes[1], "loss_drop", "Loss drop"),
        (axes[2], "final_probe_acc", "Probe accuracy"),
    ]:
        vals = mean[col].to_numpy(dtype=float)
        if col == "final_probe_acc":
            vals = 100.0 * vals
        ax.bar(x, vals, color=colors, width=0.72)
        ax.set_xticks(x, mean["method_label"], rotation=25, ha="right")
        ax.set_title(title)
    axes[0].set_ylabel("steps")
    axes[1].set_ylabel("initial-final")
    axes[2].set_ylabel("%")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"dfa_stall_summary_bars.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def smooth(values: pd.Series | np.ndarray, alpha: float = 0.06) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, arr.size):
        if np.isfinite(arr[i]):
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
        else:
            out[i] = out[i - 1]
    return out


if __name__ == "__main__":
    main()

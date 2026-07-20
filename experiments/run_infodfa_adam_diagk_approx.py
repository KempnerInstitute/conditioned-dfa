"""Adam/diagonal-K approximation tests for Conditioned DFA.

This focused synthetic benchmark tests whether the useful part of nDFA/K-nDFA
can be explained as an Adam/RMSProp-like diagonal second-moment correction.
The diagnostic separates three objects for each hidden-layer weight:

* exact per-sample local-gradient second moment, E[(delta_i h_j)^2]
* factorized KFAC diagonal, E[delta_i^2] E[h_j^2]
* Adam's running v_ij from minibatch DFA gradients

The performance comparison uses the same multi-output noisy/nuisance cells as
the main Conditioned DFA synthetic follow-ups.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dfa_multioutput_synthetic import make_multioutput_dataset  # noqa: E402
from experiments.run_dfa_synthetic import natural_precondition_gradients  # noqa: E402
from experiments.run_dfa_vision_baselines import minibatches  # noqa: E402
from infogeo.dfa import Gradients, ManualMLP, gradient_cosines, init_feedback  # noqa: E402


@dataclass(frozen=True)
class CellConfig:
    condition: str
    n_train: int
    input_noise: float
    label_noise: float


CELL_CONFIGS = {
    "clean_aligned": CellConfig("task_aligned", 4096, 0.05, 0.0),
    "nuisance_hard": CellConfig("nuisance_dominant", 512, 0.15, 0.2),
    "mixed_hard": CellConfig("mixed_context", 1024, 0.15, 0.2),
    "low_sample_noisy": CellConfig("low_sample_noisy", 512, 0.15, 0.4),
}


ADAPTIVE_METHODS = {"dfa_adam_hidden", "dfa_rmsprop_hidden"}
DAMPED_METHODS = {
    "dfa_diag_activity_sqrt",
    "dfa_diag_error_sqrt",
    "dfa_diag_k_sqrt",
    "dfa_diag_activity",
    "dfa_diag_k",
    "ndfa_activity",
    "ndfa_error",
    "kndfa",
}
FEEDBACK_METHODS = ADAPTIVE_METHODS | DAMPED_METHODS | {"dfa_sgd"}


@dataclass(frozen=True)
class RunSpec:
    cell: str
    method: str
    seed: int
    feedback_seed: int
    feedback_rank: int
    damping: float
    adaptive_lr: float


@dataclass
class AdaptiveState:
    m_w: list[torch.Tensor]
    v_w: list[torch.Tensor]
    step: int = 0


def main() -> None:
    args = parse_args()
    if args.quick:
        args.cells = ["nuisance_hard"]
        args.methods = ["bp_sgd", "dfa_sgd", "dfa_adam_hidden", "dfa_diag_k_sqrt", "ndfa_activity", "kndfa"]
        args.n_seeds = 1
        args.n_feedback_seeds = 1
        args.dampings = [0.3]
        args.adaptive_lrs = [0.003]
        args.hidden_dims = [64]
        args.n_test = 256
        args.epochs = 2
        args.eval_size = 128
        args.batch_size = 64

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs(args)
    if args.n_shards is not None:
        if args.shard_index is None:
            raise ValueError("--shard-index is required with --n-shards")
        specs = [spec for idx, spec in enumerate(specs) if idx % args.n_shards == args.shard_index]
    if args.max_runs is not None:
        specs = specs[: args.max_runs]

    print(f"Running {len(specs)} Adam/diag-K approximation specs", flush=True)
    rows: list[dict[str, float | str]] = []
    for spec_idx, spec in enumerate(specs, start=1):
        rows.extend(run_one(spec, args=args))
        if args.checkpoint_every > 0 and spec_idx % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(output_dir / "infodfa_adam_diagk_results.partial.csv", index=False)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "infodfa_adam_diagk_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}", flush=True)


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for cell in args.cells:
        if cell not in CELL_CONFIGS:
            raise ValueError(f"Unknown cell {cell!r}; choices are {sorted(CELL_CONFIGS)}")
        for method in args.methods:
            ranks = [0] if method == "bp_sgd" else args.feedback_ranks
            feedback_seeds = [0] if method == "bp_sgd" else range(args.n_feedback_seeds)
            dampings = args.dampings if method in DAMPED_METHODS else [float("nan")]
            adaptive_lrs = args.adaptive_lrs if method in ADAPTIVE_METHODS else [float("nan")]
            for seed in range(args.n_seeds):
                for feedback_rank in ranks:
                    for feedback_seed in feedback_seeds:
                        for damping in dampings:
                            for adaptive_lr in adaptive_lrs:
                                specs.append(
                                    RunSpec(
                                        cell=cell,
                                        method=method,
                                        seed=seed,
                                        feedback_seed=int(feedback_seed),
                                        feedback_rank=int(feedback_rank),
                                        damping=float(damping),
                                        adaptive_lr=float(adaptive_lr),
                                    )
                                )
    return specs


def run_one(spec: RunSpec, *, args: argparse.Namespace) -> list[dict[str, float | str]]:
    cell = CELL_CONFIGS[spec.cell]
    dataset = make_multioutput_dataset(
        condition=cell.condition,
        n_train=cell.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        n_classes=args.n_classes,
        nuisance_dim=args.nuisance_dim,
        input_noise=cell.input_noise,
        train_label_noise=cell.label_noise,
        test_label_noise=0.0,
        task_scale_override=None,
        nuisance_scale_override=None,
        seed=spec.seed,
    )
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    torch.manual_seed(10_000 + spec.seed)
    model = ManualMLP(
        input_dim=dataset.x_train.shape[1],
        hidden_dims=args.hidden_dims,
        output_dim=dataset.n_classes,
        seed=10_000 + spec.seed,
        device=device,
    )
    feedback = None
    if spec.method in FEEDBACK_METHODS:
        feedback = init_feedback(
            model,
            mode="random",
            seed=20_000 + 100 * spec.seed + spec.feedback_seed,
            scale=args.feedback_scale,
            rank=None if spec.feedback_rank <= 0 else spec.feedback_rank,
        )
    adaptive_state = make_adaptive_state(model) if spec.method in ADAPTIVE_METHODS else None

    x_train = torch.tensor(dataset.x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(dataset.y_train, dtype=torch.long, device=device)
    x_test = torch.tensor(dataset.x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(dataset.y_test, dtype=torch.long, device=device)
    rng = np.random.default_rng(30_000 + 100 * spec.seed + spec.feedback_seed)
    eval_n = min(args.eval_size, len(x_train))
    rows = []

    for epoch in range(args.epochs + 1):
        rows.append(
            evaluate(
                model,
                x_train[:eval_n],
                y_train[:eval_n],
                x_test,
                y_test,
                feedback=feedback,
                adaptive_state=adaptive_state,
                spec=spec,
                epoch=epoch,
                args=args,
            )
        )
        if epoch == args.epochs:
            break
        for batch in minibatches(len(x_train), args.batch_size, rng):
            xb = x_train[batch]
            yb = y_train[batch]
            if spec.method == "bp_sgd":
                gradients = model.bp_gradients(xb, yb)
                model.apply_gradients(gradients, lr=args.lr)
                continue
            if feedback is None:
                raise ValueError(f"{spec.method} requires feedback")
            raw = model.dfa_gradients(xb, yb, feedback)
            if spec.method in ADAPTIVE_METHODS:
                assert adaptive_state is not None
                apply_adaptive_hidden_step(
                    model,
                    raw,
                    adaptive_state,
                    kind="adam" if spec.method == "dfa_adam_hidden" else "rmsprop",
                    adaptive_lr=spec.adaptive_lr,
                    sgd_lr=args.lr,
                    beta1=args.adam_beta1,
                    beta2=args.adam_beta2,
                    eps=args.adam_eps,
                )
            else:
                gradients = transform_gradients(model, raw, xb, method=spec.method, damping=spec.damping)
                model.apply_gradients(gradients, lr=args.lr)

    print(
        f"{spec.cell:17s} {spec.method:24s} seed={spec.seed} fb={spec.feedback_seed} "
        f"rank={spec.feedback_rank} damp={format_float(spec.damping)} alr={format_float(spec.adaptive_lr)} "
        f"test={model.accuracy(x_test, y_test):.3f}",
        flush=True,
    )
    return rows


def transform_gradients(
    model: ManualMLP,
    raw: Gradients,
    x: torch.Tensor,
    *,
    method: str,
    damping: float,
) -> Gradients:
    if method in {"dfa_sgd", "dfa_adam_hidden", "dfa_rmsprop_hidden"}:
        return raw
    if method == "dfa_diag_activity_sqrt":
        return diagonal_second_moment_gradients(model, raw, x, damping=damping, left_power=0.0, right_power=0.5)
    if method == "dfa_diag_error_sqrt":
        return diagonal_second_moment_gradients(model, raw, x, damping=damping, left_power=0.5, right_power=0.0)
    if method == "dfa_diag_k_sqrt":
        return diagonal_second_moment_gradients(model, raw, x, damping=damping, left_power=0.5, right_power=0.5)
    if method == "dfa_diag_activity":
        return diagonal_second_moment_gradients(model, raw, x, damping=damping, left_power=0.0, right_power=1.0)
    if method == "dfa_diag_k":
        return diagonal_second_moment_gradients(model, raw, x, damping=damping, left_power=1.0, right_power=1.0)
    if method == "ndfa_activity":
        return natural_precondition_gradients(model, raw, x, damping=damping, mode="activity")
    if method == "ndfa_error":
        return natural_precondition_gradients(model, raw, x, damping=damping, mode="error")
    if method == "kndfa":
        return natural_precondition_gradients(model, raw, x, damping=damping, mode="kronecker")
    raise ValueError(f"Unknown method: {method}")


def diagonal_second_moment_gradients(
    model: ManualMLP,
    gradients: Gradients,
    x: torch.Tensor,
    *,
    damping: float,
    left_power: float,
    right_power: float,
) -> Gradients:
    _, activations, _ = model.forward(x)
    weights = [grad.clone() for grad in gradients.weights]
    biases = [grad.clone() for grad in gradients.biases]
    damping = max(float(damping), 1e-12)
    for layer_idx in range(model.n_hidden_layers):
        grad = gradients.weights[layer_idx]
        if left_power:
            delta_second = gradients.deltas[layer_idx].detach().pow(2).mean(dim=0).clamp_min(0.0) + damping
            grad = grad / delta_second.pow(left_power).unsqueeze(1)
        if right_power:
            activity_second = activations[layer_idx].detach().pow(2).mean(dim=0).clamp_min(0.0) + damping
            grad = grad / activity_second.pow(right_power).unsqueeze(0)
        weights[layer_idx] = grad
    return Gradients(weights, biases, gradients.deltas, gradients.loss)


def make_adaptive_state(model: ManualMLP) -> AdaptiveState:
    return AdaptiveState(
        m_w=[torch.zeros_like(weight) for weight in model.weights],
        v_w=[torch.zeros_like(weight) for weight in model.weights],
        step=0,
    )


def apply_adaptive_hidden_step(
    model: ManualMLP,
    gradients: Gradients,
    state: AdaptiveState,
    *,
    kind: str,
    adaptive_lr: float,
    sgd_lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> None:
    state.step += 1
    for layer_idx in range(len(model.weights)):
        if layer_idx < model.n_hidden_layers:
            grad = gradients.weights[layer_idx]
            state.v_w[layer_idx] = beta2 * state.v_w[layer_idx] + (1.0 - beta2) * grad.square()
            v_hat = state.v_w[layer_idx] / max(1.0 - beta2**state.step, eps)
            if kind == "adam":
                state.m_w[layer_idx] = beta1 * state.m_w[layer_idx] + (1.0 - beta1) * grad
                m_hat = state.m_w[layer_idx] / max(1.0 - beta1**state.step, eps)
                step = m_hat / (torch.sqrt(v_hat) + eps)
            elif kind == "rmsprop":
                step = grad / (torch.sqrt(v_hat) + eps)
            else:
                raise ValueError(f"Unknown adaptive kind: {kind}")
            model.weights[layer_idx] = model.weights[layer_idx] - adaptive_lr * step
        else:
            model.weights[layer_idx] = model.weights[layer_idx] - sgd_lr * gradients.weights[layer_idx]
        model.biases[layer_idx] = model.biases[layer_idx] - sgd_lr * gradients.biases[layer_idx]


def adaptive_preconditioned_gradients(
    gradients: Gradients,
    state: AdaptiveState | None,
    *,
    kind: str,
    beta1: float,
    beta2: float,
    eps: float,
    n_hidden_layers: int,
) -> Gradients:
    if state is None or state.step <= 0:
        return gradients
    weights = [grad.clone() for grad in gradients.weights]
    for layer_idx in range(n_hidden_layers):
        v_hat = state.v_w[layer_idx] / max(1.0 - beta2**state.step, eps)
        if kind == "adam":
            m_hat = state.m_w[layer_idx] / max(1.0 - beta1**state.step, eps)
            weights[layer_idx] = m_hat / (torch.sqrt(v_hat) + eps)
        else:
            weights[layer_idx] = gradients.weights[layer_idx] / (torch.sqrt(v_hat) + eps)
    return Gradients(weights, list(gradients.biases), gradients.deltas, gradients.loss)


def evaluate(
    model: ManualMLP,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    feedback: list[torch.Tensor] | None,
    adaptive_state: AdaptiveState | None,
    spec: RunSpec,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    bp = model.bp_gradients(x_eval, y_eval)
    diagnostics: dict[str, float] = {
        "raw_param_cosine": math.nan,
        "method_param_cosine": math.nan,
        "method_hidden_projected_step": math.nan,
        "method_hidden_norm_ratio": math.nan,
        "factorization_log_corr_mean": math.nan,
        "factorization_rel_error_mean": math.nan,
        "gradsq_factor_log_corr_mean": math.nan,
        "adam_factor_log_corr_mean": math.nan,
        "adam_exact_log_corr_mean": math.nan,
        "adam_diagk_update_cosine_mean": math.nan,
        "adam_m_diagk_update_cosine_mean": math.nan,
    }
    if feedback is not None:
        raw = model.dfa_gradients(x_eval, y_eval, feedback)
        if spec.method in ADAPTIVE_METHODS:
            method_grad = adaptive_preconditioned_gradients(
                raw,
                adaptive_state,
                kind="adam" if spec.method == "dfa_adam_hidden" else "rmsprop",
                beta1=args.adam_beta1,
                beta2=args.adam_beta2,
                eps=args.adam_eps,
                n_hidden_layers=model.n_hidden_layers,
            )
        else:
            method_grad = transform_gradients(
                model,
                raw,
                x_eval,
                method=spec.method,
                damping=spec.damping if np.isfinite(spec.damping) else args.diagnostic_damping,
            )
        raw_scores = gradient_cosines(bp, raw)
        method_scores = gradient_cosines(bp, method_grad)
        diagnostics.update(
            {
                "raw_param_cosine": float(raw_scores.get("param_cosine", math.nan)),
                "method_param_cosine": float(method_scores.get("param_cosine", math.nan)),
                "method_hidden_projected_step": hidden_weight_projected_step(bp, method_grad),
                "method_hidden_norm_ratio": hidden_weight_norm_ratio(method_grad, raw),
            }
        )
        diagnostics.update(
            second_moment_diagnostics(
                model,
                x_eval,
                raw,
                adaptive_state=adaptive_state,
                beta1=args.adam_beta1,
                beta2=args.adam_beta2,
                eps=args.adam_eps,
                damping=args.diagnostic_damping,
            )
        )
    return {
        "cell": spec.cell,
        "condition": CELL_CONFIGS[spec.cell].condition,
        "method": spec.method,
        "seed": float(spec.seed),
        "feedback_seed": float(spec.feedback_seed),
        "feedback_rank": float(spec.feedback_rank),
        "damping": float(spec.damping),
        "adaptive_lr": float(spec.adaptive_lr),
        "epoch": float(epoch),
        "n_train": float(CELL_CONFIGS[spec.cell].n_train),
        "n_test": float(len(x_test)),
        "input_noise": float(CELL_CONFIGS[spec.cell].input_noise),
        "train_label_noise": float(CELL_CONFIGS[spec.cell].label_noise),
        "loss": bp.loss,
        "train_eval_acc": model.accuracy(x_eval, y_eval),
        "test_acc": model.accuracy(x_test, y_test),
        **diagnostics,
    }


def second_moment_diagnostics(
    model: ManualMLP,
    x: torch.Tensor,
    raw: Gradients,
    *,
    adaptive_state: AdaptiveState | None,
    beta1: float,
    beta2: float,
    eps: float,
    damping: float,
) -> dict[str, float]:
    _, activations, _ = model.forward(x)
    factor_corrs = []
    factor_errors = []
    gradsq_corrs = []
    adam_factor_corrs = []
    adam_exact_corrs = []
    adam_diagk_cosines = []
    adam_m_diagk_cosines = []
    for layer_idx in range(model.n_hidden_layers):
        activity = activations[layer_idx].detach()
        delta = raw.deltas[layer_idx].detach()
        exact = delta.square().T @ activity.square() / max(int(activity.shape[0]), 1)
        factor = delta.square().mean(dim=0).unsqueeze(1) * activity.square().mean(dim=0).unsqueeze(0)
        factor_corrs.append(log_corr(exact, factor, eps=eps))
        factor_errors.append(relative_error(exact, factor, eps=eps))
        gradsq_corrs.append(log_corr(raw.weights[layer_idx].square(), factor, eps=eps))
        if adaptive_state is not None and adaptive_state.step > 0:
            v_hat = adaptive_state.v_w[layer_idx] / max(1.0 - beta2**adaptive_state.step, eps)
            adam_factor_corrs.append(log_corr(v_hat, factor, eps=eps))
            adam_exact_corrs.append(log_corr(v_hat, exact, eps=eps))
            diagk_step = raw.weights[layer_idx] / torch.sqrt(factor + max(float(damping), eps))
            adam_step = raw.weights[layer_idx] / (torch.sqrt(v_hat) + eps)
            adam_diagk_cosines.append(torch_cosine(adam_step, diagk_step, eps=eps))
            if adaptive_state.m_w[layer_idx].numel():
                m_hat = adaptive_state.m_w[layer_idx] / max(1.0 - beta1**adaptive_state.step, eps)
                adam_m_step = m_hat / (torch.sqrt(v_hat) + eps)
                adam_m_diagk_cosines.append(torch_cosine(adam_m_step, diagk_step, eps=eps))
    return {
        "factorization_log_corr_mean": finite_mean(factor_corrs),
        "factorization_rel_error_mean": finite_mean(factor_errors),
        "gradsq_factor_log_corr_mean": finite_mean(gradsq_corrs),
        "adam_factor_log_corr_mean": finite_mean(adam_factor_corrs),
        "adam_exact_log_corr_mean": finite_mean(adam_exact_corrs),
        "adam_diagk_update_cosine_mean": finite_mean(adam_diagk_cosines),
        "adam_m_diagk_update_cosine_mean": finite_mean(adam_m_diagk_cosines),
    }


def hidden_weight_projected_step(reference: Gradients, estimate: Gradients) -> float:
    numerator = 0.0
    denominator = 0.0
    for ref, est in zip(reference.weights[:-1], estimate.weights[:-1]):
        numerator += float(torch.sum(ref * est).detach().cpu().item())
        denominator += float(torch.sum(ref * ref).detach().cpu().item())
    return numerator / max(denominator, 1e-12)


def hidden_weight_norm_ratio(numerator_grad: Gradients, denominator_grad: Gradients) -> float:
    num = sum(float(torch.sum(w * w).detach().cpu().item()) for w in numerator_grad.weights[:-1])
    den = sum(float(torch.sum(w * w).detach().cpu().item()) for w in denominator_grad.weights[:-1])
    return float(np.sqrt(num / max(den, 1e-12)))


def log_corr(a: torch.Tensor, b: torch.Tensor, *, eps: float) -> float:
    x = torch.log(torch.nan_to_num(a.detach().flatten().float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(eps))
    y = torch.log(torch.nan_to_num(b.detach().flatten().float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(eps))
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 3:
        return math.nan
    x = x[mask] - x[mask].mean()
    y = y[mask] - y[mask].mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if float(denom.detach().cpu().item()) <= eps:
        return math.nan
    return float((torch.dot(x, y) / denom).detach().cpu().item())


def relative_error(a: torch.Tensor, b: torch.Tensor, *, eps: float) -> float:
    denom = torch.linalg.norm(a.detach().float()).clamp_min(eps)
    return float((torch.linalg.norm((a - b).detach().float()) / denom).detach().cpu().item())


def torch_cosine(a: torch.Tensor, b: torch.Tensor, *, eps: float) -> float:
    x = a.detach().flatten().float()
    y = b.detach().flatten().float()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if float(denom.detach().cpu().item()) <= eps:
        return math.nan
    return float((torch.dot(x, y) / denom).detach().cpu().item())


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(arr.mean())


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="results/infodfa_adam_diagk_approx_v1")
    parser.add_argument("--cells", nargs="+", default=["clean_aligned", "nuisance_hard", "mixed_hard", "low_sample_noisy"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "bp_sgd",
            "dfa_sgd",
            "dfa_adam_hidden",
            "dfa_diag_activity_sqrt",
            "dfa_diag_error_sqrt",
            "dfa_diag_k_sqrt",
            "ndfa_activity",
            "ndfa_error",
            "kndfa",
        ],
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-feedback-seeds", type=int, default=2)
    parser.add_argument("--feedback-ranks", type=int, nargs="+", default=[0])
    parser.add_argument("--dampings", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--adaptive-lrs", type=float, nargs="+", default=[0.001, 0.003])
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--n-classes", type=int, default=8)
    parser.add_argument("--nuisance-dim", type=int, default=24)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--diagnostic-damping", type=float, default=0.3)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--n-shards", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main()

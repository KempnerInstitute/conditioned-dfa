"""Does conditioning change DFA's align-then-memorize dynamics?

Refinetti et al. (2021) show random-feedback learning proceeds in two phases:
an alignment phase (forward weights adapt to the fixed feedback) followed by a
memorization phase. The Info-DFA paper predicts that right-multiplying the DFA
update by the damped inverse activity second moment (nDFA / K-nDFA) changes
update *geometry* while leaving the random-feedback *alignment* dynamics
untouched, and that the E_B[Pi] gap between conditioned and raw DFA is largest
during the alignment phase.

This script measures both claims directly on the paper's two focused synthetic
cells (nuisance-dominant hard cell and clean task-aligned cell, factor-ablation
protocol). For each method (DFA, nDFA, K-nDFA) it trains the standard ManualMLP
and logs, on a fine early / per-epoch late time grid:

- weight_align_l{k}: cos(vec(M_k), vec(B_k)) where M_k is the linearized
  output-to-hidden forward chain (W_L ... W_{k+1}) and B_k the fixed DFA
  feedback -- the DFA analogue of Refinetti's weight-alignment order parameter.
  This quantity is method-comparable because it depends only on the forward
  weights and the (shared, seed-matched) feedback draw.
- pi_weight_l{k} / pi_weight_mean: projected BP-step ratio in weight-gradient
  space, Pi(g, g*) = <g, g*> / ||g*||^2 (paper eq. in App. Pi section), where g
  includes the method's preconditioning. This separates geometry at *matched*
  weights.
- projected_step_ratio_mean: the delta-space Pi logged by the main sweeps
  (continuity with existing aggregates).
- param_cosine / activity cosines, loss, test accuracy.

Nothing in the training path is modified: datasets, initialization, feedback
draws, learning rate, damping, and update order are imported from the standard
runner so trajectories are seed-matched with the factor-ablation runs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dfa_multioutput_synthetic import (  # noqa: E402
    compute_gradients,
    initialize_feedback,
    make_multioutput_dataset,
)
from experiments.run_dfa_vision_baselines import minibatches  # noqa: E402
from infogeo.dfa import ManualMLP, _linear_backprop_matrix, gradient_cosines  # noqa: E402
from infogeo.noise_correlation import noise_correlation_diagnostics  # noqa: E402


# Focused cells from slurm/infodfa_factor_ablation_synthetic.sbatch.
CELLS = {
    "nuisance_hard": dict(condition="nuisance_dominant", n_train=512, label_noise=0.2, input_noise=0.15),
    "clean_aligned": dict(condition="task_aligned", n_train=4096, label_noise=0.0, input_noise=0.05),
}
METHODS = ["dfa_random", "ndfa_random", "ndfa_random_kronecker"]


def weight_alignment(model: ManualMLP, feedback) -> dict[str, float]:
    """Refinetti-style alignment: forward output-to-hidden chain vs fixed feedback."""

    out: dict[str, float] = {}
    values = []
    for layer_idx in range(model.n_hidden_layers):
        chain = _linear_backprop_matrix(model, layer_idx)
        fb = feedback[layer_idx].detach().cpu().numpy()
        denom = max(float(np.linalg.norm(chain) * np.linalg.norm(fb)), 1e-12)
        cos = float(np.sum(chain * fb) / denom)
        out[f"weight_align_l{layer_idx + 1}"] = cos
        values.append(cos)
    out["weight_align_mean"] = float(np.mean(values))
    return out


def weight_space_pi(bp, local) -> dict[str, float]:
    """Pi(g, g*) = <g, g*> / ||g*||^2 on hidden weight gradients."""

    out: dict[str, float] = {}
    values = []
    for layer_idx in range(len(bp.weights) - 1):
        g_star = bp.weights[layer_idx].flatten()
        g = local.weights[layer_idx].flatten()
        pi = float((torch.dot(g, g_star) / torch.dot(g_star, g_star).clamp_min(1e-24)).item())
        out[f"pi_weight_l{layer_idx + 1}"] = pi
        values.append(pi)
    out["pi_weight_mean"] = float(np.mean(values))
    return out


def eval_point(model, x_eval, y_eval, x_test, y_test, *, method, feedback, args) -> dict[str, float]:
    bp = model.bp_gradients(x_eval, y_eval)
    local = compute_gradients(model, x_eval, y_eval, method=method, feedback=feedback, args=args)
    row: dict[str, float] = {
        "loss": bp.loss,
        "test_acc": model.accuracy(x_test, y_test),
    }
    row.update(weight_alignment(model, feedback))
    row.update(weight_space_pi(bp, local))
    row.update({k: float(v) for k, v in gradient_cosines(bp, local).items() if np.isscalar(v)})
    row.update(noise_correlation_diagnostics(bp, local, None))
    return row


def run_one(cell: str, method: str, seed: int, feedback_seed: int, args: argparse.Namespace) -> list[dict]:
    spec = CELLS[cell]
    dataset = make_multioutput_dataset(
        condition=spec["condition"],
        n_train=spec["n_train"],
        n_test=args.n_test,
        input_dim=64,
        n_classes=8,
        nuisance_dim=24,
        input_noise=spec["input_noise"],
        train_label_noise=spec["label_noise"],
        test_label_noise=0.0,
        task_scale_override=None,
        nuisance_scale_override=None,
        seed=seed,
    )
    model = ManualMLP(
        input_dim=dataset.x_train.shape[1],
        hidden_dims=args.hidden_dims,
        output_dim=dataset.n_classes,
        seed=10_000 + seed,
        device="cpu",
    )
    x_train = torch.tensor(dataset.x_train)
    y_train = torch.tensor(dataset.y_train)
    x_test = torch.tensor(dataset.x_test)
    y_test = torch.tensor(dataset.y_test)
    feedback = initialize_feedback(
        model, method=method, seed=seed, feedback_seed=feedback_seed, feedback_rank=0, args=args
    )
    rng = np.random.default_rng(30_000 + 100 * seed + feedback_seed)
    eval_n = min(args.eval_size, len(x_train))

    steps_per_epoch = max(1, int(np.ceil(len(x_train) / args.batch_size)))
    # Fine grid (quarter-epoch) during the first `fine_epochs`, per-epoch after.
    fine_stride = max(1, steps_per_epoch // 4)
    eval_steps = {0}
    for step in range(steps_per_epoch * args.epochs + 1):
        if step < args.fine_epochs * steps_per_epoch and step % fine_stride == 0:
            eval_steps.add(step)
        if step % steps_per_epoch == 0:
            eval_steps.add(step)

    base = {
        "cell": cell,
        "condition": dataset.condition,
        "method": method,
        "seed": seed,
        "feedback_seed": feedback_seed,
    }
    rows = []
    step = 0
    for epoch in range(args.epochs):
        if step in eval_steps:
            rows.append({**base, "step": step, "epoch": step / steps_per_epoch,
                         **eval_point(model, x_train[:eval_n], y_train[:eval_n], x_test, y_test,
                                      method=method, feedback=feedback, args=args)})
        for batch in minibatches(len(x_train), args.batch_size, rng):
            gradients = compute_gradients(model, x_train[batch], y_train[batch],
                                          method=method, feedback=feedback, args=args)
            model.apply_gradients(gradients, lr=args.lr)
            step += 1
            if step in eval_steps and step % steps_per_epoch != 0:
                rows.append({**base, "step": step, "epoch": step / steps_per_epoch,
                             **eval_point(model, x_train[:eval_n], y_train[:eval_n], x_test, y_test,
                                          method=method, feedback=feedback, args=args)})
    rows.append({**base, "step": step, "epoch": float(args.epochs),
                 **eval_point(model, x_train[:eval_n], y_train[:eval_n], x_test, y_test,
                              method=method, feedback=feedback, args=args)})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/infodfa_alignment_dynamics_v1")
    parser.add_argument("--cells", nargs="+", default=list(CELLS))
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit seed list (overrides --n-seeds).")
    parser.add_argument("--n-feedback-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--fine-epochs", type=int, default=5)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--natural-damping", type=float, default=0.3)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--nc-init", default="zero")
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))
    for cell in args.cells:
        for method in args.methods:
            for seed in seeds:
                for feedback_seed in range(args.n_feedback_seeds):
                    rows.extend(run_one(cell, method, seed, feedback_seed, args))
                    print(f"{cell:14s} {method:22s} seed={seed} fb={feedback_seed} "
                          f"elapsed={time.time() - t0:7.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(output_dir / "alignment_dynamics.partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "alignment_dynamics.csv", index=False)
    print(f"Saved {output_dir / 'alignment_dynamics.csv'} ({len(df)} rows, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

"""Validation-safe activity/error/K-nDFA test in a ReLU vision MLP.

This is an architectural and loss-function replication of the tanh DFA-Stall
experiment: a two-hidden-layer ReLU MLP is trained with multiclass softmax
cross-entropy.  Activity and error dampings can be selected independently on a
fixed training-validation split; conditioned hidden weight gradients are
layerwise norm-matched to raw DFA.  Test evaluation is opt-in so development
runs need not touch the test set.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dfa_factorial_synthetic import METHOD_LABEL, condition_gradients  # noqa: E402
from infogeo.dfa import ManualMLP, init_feedback  # noqa: E402


METHODS = ("dfa", "ndfa", "endfa", "kndfa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/dfa_relu_vision_threefactor_v1")
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--feedback-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--activity-damping", type=float, default=0.3)
    parser.add_argument("--error-damping", type=float, default=3.0)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--validation-seed", type=int, default=86420)
    parser.add_argument("--probe-n", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(args)

    rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        for feedback_seed in args.feedback_seeds:
            for method in args.methods:
                rows.extend(
                    run_one(
                        args,
                        train_x,
                        train_y,
                        val_x,
                        val_y,
                        test_x,
                        test_y,
                        seed=seed,
                        feedback_seed=feedback_seed,
                        method=method,
                        device=device,
                    )
                )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "relu_vision_results.csv", index=False)
    endpoints = df.sort_values("step").groupby(["method", "seed", "feedback_seed"], as_index=False).tail(1)
    endpoints.to_csv(output_dir / "relu_vision_endpoints.csv", index=False)
    write_report(endpoints, output_dir, args)
    print(f"wrote {output_dir}")


def load_dataset(args: argparse.Namespace):
    from torchvision import datasets

    cls = datasets.MNIST if args.dataset == "mnist" else datasets.FashionMNIST
    train = cls(root=args.data_dir, train=True, download=True)
    test = cls(root=args.data_dir, train=False, download=True)
    full_x = train.data.float().div(255.0).view(-1, 784)
    full_y = torch.as_tensor(train.targets, dtype=torch.long)
    test_x = test.data.float().div(255.0).view(-1, 784)
    test_y = torch.as_tensor(test.targets, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(args.validation_seed)
    order = torch.randperm(full_x.shape[0], generator=generator)
    n_val = min(args.validation_size, full_x.shape[0] - 1)
    val_idx, train_idx = order[:n_val], order[n_val:]
    return full_x[train_idx], full_y[train_idx], full_x[val_idx], full_y[val_idx], test_x, test_y


def run_one(
    args: argparse.Namespace,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    seed: int,
    feedback_seed: int,
    method: str,
    device: str,
) -> list[dict[str, float | int | str]]:
    model = ManualMLP(784, args.hidden_dims, 10, seed=10_000 + seed, device=device)
    feedback = init_feedback(
        model,
        mode="random",
        seed=20_000 + 100 * seed + feedback_seed,
        scale=args.feedback_scale,
    )
    probe_n = min(args.probe_n, val_x.shape[0])
    probe_x = val_x[:probe_n].to(device)
    probe_y = val_y[:probe_n].to(device)
    rng = torch.Generator(device="cpu").manual_seed(30_000 + 100 * seed + feedback_seed)
    rows = []
    for step in range(1, args.total_steps + 1):
        idx = torch.randint(0, train_x.shape[0], (args.batch_size,), generator=rng)
        xb = train_x[idx].to(device)
        yb = train_y[idx].to(device)
        raw = model.dfa_gradients(xb, yb, feedback)
        gradients = condition_gradients(model, raw, xb, method=method, args=args)
        model.apply_gradients(gradients, lr=args.lr)
        if step == 1 or step % args.eval_every == 0 or step == args.total_steps:
            val_loss, val_acc = evaluate(model, probe_x, probe_y)
            test_loss = test_acc = float("nan")
            if step == args.total_steps and args.evaluate_test:
                test_loss, test_acc = evaluate(model, test_x.to(device), test_y.to(device))
            rows.append(
                {
                    "dataset": args.dataset,
                    "method": method,
                    "method_label": METHOD_LABEL[method],
                    "seed": seed,
                    "feedback_seed": feedback_seed,
                    "step": step,
                    "activity_damping": float(args.activity_damping),
                    "error_damping": float(args.error_damping),
                    "train_batch_loss": float(raw.loss),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )
    print(
        f"dataset={args.dataset:13s} method={method:5s} seed={seed:3d} fb={feedback_seed} "
        f"val={rows[-1]['val_acc']:.3f}",
        flush=True,
    )
    return rows


@torch.no_grad()
def evaluate(model: ManualMLP, x: torch.Tensor, y: torch.Tensor, chunk_size: int = 2048) -> tuple[float, float]:
    losses = []
    correct = total = 0
    for start in range(0, x.shape[0], chunk_size):
        xb, yb = x[start : start + chunk_size], y[start : start + chunk_size]
        logits, _, _ = model.forward(xb)
        losses.append((float(F.cross_entropy(logits, yb).item()), int(xb.shape[0])))
        correct += int((logits.argmax(1) == yb).sum().item())
        total += int(yb.numel())
    return float(np.average([item[0] for item in losses], weights=[item[1] for item in losses])), correct / total


def write_report(endpoints: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    summary = endpoints.groupby(["method", "method_label"], as_index=False).agg(
        val_acc_mean=("val_acc", "mean"),
        val_acc_sem=("val_acc", sem),
        val_loss_mean=("val_loss", "mean"),
        test_acc_mean=("test_acc", "mean"),
        test_loss_mean=("test_loss", "mean"),
        n=("val_acc", "count"),
    )
    lines = [
        f"# {args.dataset} ReLU/softmax three-factor experiment",
        "",
        f"Architecture {args.hidden_dims}; {args.total_steps} updates; activity/error damping "
        f"{args.activity_damping:g}/{args.error_damping:g}; layerwise hidden-gradient norm matching.",
        "",
        summary.to_markdown(index=False, floatfmt=".5f"),
    ]
    (output_dir / "relu_vision_summary.md").write_text("\n".join(lines) + "\n")


def sem(values: pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size <= 1:
        return 0.0
    return float(array.std(ddof=1) / math.sqrt(array.size))


if __name__ == "__main__":
    main()

"""DFA/FA/NDFA baselines on standard vision datasets."""

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

from experiments.run_dfa_synthetic import (
    feedback_mode_from_method,
    method_has_dynamic_feedback,
    natural_mode_from_method,
    natural_precondition_gradients,
)
from infogeo.dfa import (
    ManualMLP,
    gradient_cosines,
    init_fa_feedback,
    init_feedback,
    local_pca_tangent_spaces,
    rayleigh_quotient_scores,
    tangent_projected_cosines,
)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_train = 1024
        args.n_test = 512
        args.epochs = 2
        args.hidden_dims = [128]
        args.methods = ["bp", "dfa_random", "ndfa_random", "drtp_random"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, test_x, test_y = load_vision_dataset(args)
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
    csv_path = output_dir / "dfa_vision_results.csv"
    df.to_csv(csv_path, index=False)
    plot_results(df, output_dir)
    print(df.sort_values("epoch").groupby(["dataset", "method", "seed", "feedback_seed"]).tail(1).groupby("method")["test_acc"].mean().to_string())
    print(f"\nSaved results to {csv_path}")


def load_vision_dataset(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install torchvision to run vision baselines") from exc

    root = Path(args.data_dir)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_name = args.dataset.lower()
    if dataset_name == "mnist":
        cls = datasets.MNIST
        args.n_classes = 10
    elif dataset_name == "fashion_mnist":
        cls = datasets.FashionMNIST
        args.n_classes = 10
    elif dataset_name == "cifar10":
        cls = datasets.CIFAR10
        args.n_classes = 10
    elif dataset_name == "cifar100":
        cls = datasets.CIFAR100
        args.n_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    train = cls(root=str(root), train=True, download=args.download, transform=transform)
    test = cls(root=str(root), train=False, download=args.download, transform=transform)
    train_x, train_y = dataset_to_tensors(train, args.n_train, seed=args.seed)
    test_x, test_y = dataset_to_tensors(test, args.n_test, seed=args.seed + 1)
    train_x, test_x = normalize_train_test(train_x, test_x)
    return train_x, train_y, test_x, test_y


def dataset_to_tensors(dataset, n_samples: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    xs = []
    ys = []
    for i in idx:
        x, y = dataset[int(i)]
        xs.append(x.reshape(-1))
        ys.append(int(y))
    return torch.stack(xs).float(), torch.tensor(ys, dtype=torch.long)


def normalize_train_test(train_x: torch.Tensor, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


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

    feedback = None
    if method != "bp":
        feedback = initialize_feedback(model, method=method, seed=seed, feedback_seed=feedback_seed, args=args)
    rng = np.random.default_rng(30_000 + 100 * seed + feedback_seed)
    rows = []
    eval_n = min(args.eval_size, train_x.shape[0])
    for epoch in range(args.epochs + 1):
        rows.append(evaluate(model, train_x[:eval_n], train_y[:eval_n], test_x, test_y, method, feedback, seed, feedback_seed, epoch, args))
        if epoch == args.epochs:
            break
        if feedback is not None and method_has_dynamic_feedback(method):
            feedback = init_feedback(
                model,
                mode=feedback_mode_from_method(method),
                seed=20_000 + 100 * seed + feedback_seed,
                scale=args.feedback_scale,
                rank=None if args.feedback_rank <= 0 else args.feedback_rank,
            )
        for batch in minibatches(train_x.shape[0], args.batch_size, rng):
            xb = train_x[batch]
            yb = train_y[batch]
            if method == "bp":
                gradients = model.bp_gradients(xb, yb)
            else:
                assert feedback is not None
                if method.startswith("drtp_"):
                    gradients = model.target_projection_gradients(xb, yb, feedback)
                elif method.startswith("fa_"):
                    gradients = model.fa_gradients(xb, yb, feedback)
                else:
                    gradients = model.dfa_gradients(xb, yb, feedback)
                if method.startswith("ndfa_"):
                    gradients = natural_precondition_gradients(
                        model,
                        gradients,
                        xb,
                        damping=args.natural_damping,
                        mode=natural_mode_from_method(method),
                    )
            model.apply_gradients(gradients, lr=args.lr)
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
    if method.startswith("fa_"):
        return init_fa_feedback(
            model,
            seed=20_000 + 100 * seed + feedback_seed,
            scale=args.feedback_scale,
            rank=None if args.feedback_rank <= 0 else args.feedback_rank,
        )
    return init_feedback(
        model,
        mode=feedback_mode_from_method(method),
        seed=20_000 + 100 * seed + feedback_seed,
        scale=args.feedback_scale,
        rank=None if args.feedback_rank <= 0 else args.feedback_rank,
    )


def evaluate(
    model: ManualMLP,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    method: str,
    feedback,
    seed: int,
    feedback_seed: int,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    bp = model.bp_gradients(x_eval, y_eval)
    _, activations, _ = model.forward(x_eval)
    rayleigh = [
        np.mean(
            rayleigh_quotient_scores(
                activations[layer_idx].detach().cpu().numpy(),
                model.weights[layer_idx].detach().cpu().numpy(),
            )
        )
        for layer_idx in range(model.n_hidden_layers)
    ]
    alignments = {"param_cosine": np.nan}
    if feedback is not None:
        if method.startswith("drtp_"):
            local = model.target_projection_gradients(x_eval, y_eval, feedback)
        elif method.startswith("fa_"):
            local = model.fa_gradients(x_eval, y_eval, feedback)
        else:
            local = model.dfa_gradients(x_eval, y_eval, feedback)
        alignments = gradient_cosines(bp, local)
        hidden = [h.detach().cpu().numpy() for h in model.hidden_activations(x_eval)]
        tangents = [
            local_pca_tangent_spaces(h, n_neighbors=args.local_pca_neighbors, rank=args.local_pca_rank)
            for h in hidden
        ]
        tangent_alignments = tangent_projected_cosines(bp, local, tangents)
        finite_tangent = [value for value in tangent_alignments.values() if np.isfinite(value)]
        local_tangent_cosine = float(np.mean(finite_tangent)) if finite_tangent else np.nan
    else:
        local_tangent_cosine = np.nan
    return {
        "dataset": args.dataset,
        "method": method,
        "seed": float(seed),
        "feedback_seed": float(feedback_seed),
        "feedback_rank": float(args.feedback_rank),
        "epoch": float(epoch),
        "loss": bp.loss,
        "train_eval_acc": model.accuracy(x_eval, y_eval),
        "test_acc": model.accuracy(x_test, y_test),
        "param_cosine": float(alignments.get("param_cosine", np.nan)),
        "local_pca_tangent_cosine": local_tangent_cosine,
        "weight_rayleigh": float(np.mean(rayleigh)) if rayleigh else np.nan,
    }


def minibatches(n_samples: int, batch_size: int, rng: np.random.Generator):
    indices = rng.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        yield indices[start : start + batch_size]


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method, group in df.groupby("method"):
        curve = group.groupby("epoch")["test_acc"].mean()
        axes[0].plot(curve.index, curve.values, marker="o", label=method)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    final = df.sort_values("epoch").groupby(["method", "seed", "feedback_seed"]).tail(1)
    by_method = final.groupby("method")["test_acc"].mean().sort_values()
    axes[1].barh(by_method.index, by_method.values)
    axes[1].set_xlabel("Final test accuracy")
    fig.savefig(output_dir / "dfa_vision_summary.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="results/dfa_vision")
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist", "cifar10", "cifar100"])
    parser.add_argument("--data-dir", default="data/torchvision")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["bp", "dfa_random", "dfa_bp_aligned_dynamic", "ndfa_random", "drtp_random"])
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--n-feedback-seeds", type=int, default=2)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--feedback-rank", type=int, default=0)
    parser.add_argument("--natural-damping", type=float, default=1e-2)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--local-pca-neighbors", type=int, default=24)
    parser.add_argument("--local-pca-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Convolutional DFA/nDFA/K-nDFA baselines on harder vision tasks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dfa_vision_baselines import minibatches  # noqa: E402
from infogeo.analysis import dataframe_to_markdown, predictor_scores, write_markdown_report  # noqa: E402
from infogeo.conv_dfa import (  # noqa: E402
    ManualConvNet,
    apply_conv_local_head_gradients,
    conv_local_loss_gradients,
    conv_gradient_cosines,
    conv_hidden_weight_norm_ratio,
    conv_weight_norm_ratio,
    init_conv_feedback,
    init_conv_local_heads,
    natural_precondition_conv_gradients,
    norm_match_conv_gradients,
    spatial_kronecker_conv_gradients,
)


LOCAL_LOSS_METHODS = {"local_loss"}
IMAGENET_DATASETS = {"imagenet", "imagenet1k", "imagenet100", "imagenet_subset"}


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_train = 512
        args.n_test = 256
        args.epochs = 1
        args.channels = [8, 16]
        args.strides = [1, 2]
        args.n_seeds = 1
        args.n_feedback_seeds = 1
        args.methods = ["bp", "dfa_random", "ndfa_random_kronecker", "drtp_random", "local_loss"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, test_x, test_y, input_shape, n_classes = load_image_dataset(args)
    args.n_classes = n_classes
    train_y = corrupt_labels(train_y, n_classes=n_classes, noise=args.label_noise, seed=args.seed + 101)
    test_y = corrupt_labels(test_y, n_classes=n_classes, noise=args.test_label_noise, seed=args.seed + 202)
    rows = []
    for seed in range(args.n_seeds):
        for method in args.methods:
            feedback_seeds = [0] if method == "bp" or method in LOCAL_LOSS_METHODS else range(args.n_feedback_seeds)
            for feedback_seed in feedback_seeds:
                rows.extend(run_one(train_x, train_y, test_x, test_y, input_shape, method=method, seed=seed, feedback_seed=int(feedback_seed), args=args))
    df = pd.DataFrame(rows)
    csv_path = output_dir / "dfa_convnet_results.csv"
    df.to_csv(csv_path, index=False)
    if not args.skip_analysis:
        plot_results(df, output_dir)
        write_report(df, output_dir)
    print(final_summary(df).to_string(index=False))
    print(f"\nSaved {csv_path}")


def load_image_dataset(args: argparse.Namespace):
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install torchvision to run convnet baselines") from exc

    dataset_name = args.dataset.lower()
    root = Path(args.data_dir)
    if dataset_name in {"mnist", "fashion_mnist"}:
        transform = transforms.Compose([transforms.ToTensor()])
        cls = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
        n_classes = 10
    elif dataset_name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        cls = datasets.CIFAR10
        n_classes = 10
    elif dataset_name == "cifar100":
        transform = transforms.Compose([transforms.ToTensor()])
        cls = datasets.CIFAR100
        n_classes = 100
    elif dataset_name in IMAGENET_DATASETS:
        return load_imagefolder_imagenet(args, transforms)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    train = cls(root=str(root), train=True, download=args.download, transform=transform)
    test = cls(root=str(root), train=False, download=args.download, transform=transform)
    train_x, train_y = dataset_to_tensors(train, args.n_train, seed=args.seed)
    test_x, test_y = dataset_to_tensors(test, args.n_test, seed=args.seed + 1)
    train_x, test_x = normalize_by_channel(train_x, test_x)
    return train_x, train_y, test_x, test_y, tuple(train_x.shape[1:]), n_classes


def load_imagefolder_imagenet(args: argparse.Namespace, transforms):
    from torchvision import datasets

    root = resolve_imagenet_root(args)
    train_dir, val_dir = find_imagenet_splits(root)
    image_size = int(args.image_size)
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    train = datasets.ImageFolder(str(train_dir), transform=transform)
    val = datasets.ImageFolder(str(val_dir), transform=transform)
    n_classes = requested_imagenet_classes(args)
    if n_classes < len(train.classes):
        train, val = subset_imagefolder_classes(train, val, n_classes=n_classes, seed=args.class_subset_seed)
    else:
        n_classes = len(train.classes)
    train_x, train_y = dataset_to_tensors(train, args.n_train, seed=args.seed, n_classes=n_classes, balanced=args.balanced_subsample)
    test_x, test_y = dataset_to_tensors(val, args.n_test, seed=args.seed + 1, n_classes=n_classes, balanced=args.balanced_subsample)
    train_x, test_x = normalize_by_channel(train_x, test_x)
    return train_x, train_y, test_x, test_y, tuple(train_x.shape[1:]), n_classes


def resolve_imagenet_root(args: argparse.Namespace) -> Path:
    candidates = [
        args.imagenet_root,
        os.environ.get("IMAGENET_ROOT"),
        os.environ.get("ILSVRC_ROOT"),
        "data/imagenet",
        "data/ILSVRC2012",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    raise FileNotFoundError(
        "ImageNet root not found. Set --imagenet-root or IMAGENET_ROOT to a directory "
        "containing train/ and val/ ImageFolder splits."
    )


def find_imagenet_splits(root: Path) -> tuple[Path, Path]:
    train_names = ["train", "Train", "ILSVRC2012_img_train"]
    val_names = ["val", "validation", "Validation", "ILSVRC2012_img_val"]
    train_dir = next((root / name for name in train_names if (root / name).is_dir()), None)
    val_dir = next((root / name for name in val_names if (root / name).is_dir()), None)
    if train_dir is None or val_dir is None:
        raise FileNotFoundError(f"Could not find ImageNet train/val ImageFolder splits under {root}")
    return train_dir, val_dir


def requested_imagenet_classes(args: argparse.Namespace) -> int:
    if args.imagenet_classes > 0:
        return int(args.imagenet_classes)
    dataset_name = args.dataset.lower()
    if dataset_name == "imagenet100":
        return 100
    if dataset_name == "imagenet_subset":
        return 100
    return 1000


class RemappedImageFolderSubset(torch.utils.data.Dataset):
    def __init__(self, base, old_to_new: dict[int, int]):
        self.base = base
        self.old_to_new = dict(old_to_new)
        self.indices = [idx for idx, (_, label) in enumerate(base.samples) if int(label) in self.old_to_new]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        x, y = self.base[self.indices[item]]
        return x, self.old_to_new[int(y)]


def subset_imagefolder_classes(train, val, *, n_classes: int, seed: int):
    rng = np.random.default_rng(seed)
    class_ids = np.arange(len(train.classes))
    selected = np.sort(rng.choice(class_ids, size=min(n_classes, len(class_ids)), replace=False))
    old_to_new = {int(old): int(new) for new, old in enumerate(selected)}
    return RemappedImageFolderSubset(train, old_to_new), RemappedImageFolderSubset(val, old_to_new)


def dataset_to_tensors(dataset, n_samples: int, *, seed: int, n_classes: int | None = None, balanced: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    if balanced and n_classes is not None and n_classes > 1:
        labels = np.array([dataset_label_at(dataset, int(i)) for i in range(len(dataset))])
        per_class = max(int(np.ceil(n_samples / n_classes)), 1)
        selected: list[int] = []
        for cls in range(n_classes):
            cls_idx = np.flatnonzero(labels == cls)
            if cls_idx.size == 0:
                continue
            take = min(per_class, cls_idx.size)
            selected.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
        idx = np.array(selected, dtype=int)
        if idx.size > n_samples:
            idx = rng.choice(idx, size=n_samples, replace=False)
        rng.shuffle(idx)
    else:
        idx = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    xs = []
    ys = []
    for i in idx:
        x, y = dataset[int(i)]
        xs.append(x)
        ys.append(int(y))
    return torch.stack(xs).float(), torch.tensor(ys, dtype=torch.long)


def dataset_label_at(dataset, index: int) -> int:
    if isinstance(dataset, RemappedImageFolderSubset):
        old_label = int(dataset.base.samples[dataset.indices[index]][1])
        return int(dataset.old_to_new[old_label])
    if hasattr(dataset, "samples"):
        return int(dataset.samples[index][1])
    if hasattr(dataset, "targets"):
        return int(dataset.targets[index])
    return int(dataset[index][1])


def normalize_by_channel(train_x: torch.Tensor, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=(0, 2, 3), keepdim=True)
    std = train_x.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def run_one(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    input_shape: tuple[int, int, int],
    *,
    method: str,
    seed: int,
    feedback_seed: int,
    args: argparse.Namespace,
) -> list[dict[str, float | str]]:
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model = ManualConvNet(
        input_shape=input_shape,
        output_dim=args.n_classes,
        channels=args.channels,
        strides=args.strides,
        kernels=args.kernels,
        paddings=args.paddings,
        fc_hidden=args.fc_hidden,
        seed=10_000 + seed,
        device=device,
    )
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    feedback = None
    local_heads = None
    if method in LOCAL_LOSS_METHODS:
        local_heads = init_conv_local_heads(model, seed=40_000 + seed)
    elif method != "bp":
        feedback = init_conv_feedback(
            model,
            seed=20_000 + 100 * seed + feedback_seed,
            scale=args.feedback_scale,
            rank=None if args.feedback_rank <= 0 else args.feedback_rank,
            mode=args.feedback_mode,
            normalization=args.feedback_normalization,
        )
    rng = np.random.default_rng(30_000 + 100 * seed + feedback_seed)
    opt_state: dict[str, object] = {"step": 0}
    rows = []
    eval_n = min(args.eval_size, len(train_x))
    for epoch in range(args.epochs + 1):
        rows.append(evaluate(model, train_x[:eval_n], train_y[:eval_n], test_x, test_y, method, feedback, local_heads, seed, feedback_seed, epoch, args))
        if epoch == args.epochs:
            break
        for batch in minibatches(len(train_x), args.batch_size, rng):
            xb = train_x[batch]
            yb = train_y[batch]
            gradients, head_gradients = compute_gradients(model, xb, yb, method=method, feedback=feedback, local_heads=local_heads, args=args)
            if args.gradient_clip > 0:
                gradients = clip_gradients(gradients, args.gradient_clip)
            apply_update(model, gradients, args=args, state=opt_state)
            if local_heads is not None and head_gradients is not None:
                apply_conv_local_head_gradients(local_heads, head_gradients, lr=args.lr)
    return rows


def compute_gradients(model: ManualConvNet, x: torch.Tensor, y: torch.Tensor, *, method: str, feedback, local_heads, args: argparse.Namespace):
    if method == "bp":
        return model.bp_gradients(x, y), None
    if method in LOCAL_LOSS_METHODS:
        if local_heads is None:
            raise ValueError(f"{method} requires local heads")
        gradients, head_gradients = conv_local_loss_gradients(model, x, y, local_heads)
        return gradients, head_gradients
    if feedback is None:
        raise ValueError(f"{method} requires feedback")
    if method.startswith("drtp_"):
        gradients = model.target_projection_gradients(x, y, feedback)
    else:
        gradients = model.dfa_gradients(x, y, feedback)
    if method == "ndfa_random":
        gradients = natural_precondition_conv_gradients(model, gradients, x, damping=args.natural_damping, mode="activity")
    elif method == "ndfa_random_kronecker":
        gradients = natural_precondition_conv_gradients(model, gradients, x, damping=args.natural_damping, mode="kronecker")
    elif method == "ndfa_spatial_kron":
        gradients = spatial_kronecker_conv_gradients(model, gradients, x, damping=args.natural_damping)
    elif method == "dfa_random_normmatch":
        gradients = norm_match_conv_gradients(model, gradients, x, y)
    elif method not in {"dfa_random", "drtp_random"}:
        raise ValueError(f"Unknown method: {method}")
    return gradients, None


def evaluate(
    model: ManualConvNet,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    method: str,
    feedback,
    local_heads,
    seed: int,
    feedback_seed: int,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    bp = model.bp_gradients(x_eval, y_eval)
    diagnostics = {
        "param_cosine": np.nan,
        "hidden_param_cosine": np.nan,
        "output_param_cosine": np.nan,
        "activity_cosine_mean": np.nan,
        "projected_weight_step": np.nan,
        "raw_projected_weight_step": np.nan,
        "hidden_projected_weight_step": np.nan,
        "raw_hidden_projected_weight_step": np.nan,
        "output_projected_weight_step": np.nan,
        "hidden_weight_norm_ratio": np.nan,
        "kndfa_weight_norm_ratio": np.nan,
        "kndfa_hidden_weight_norm_ratio": np.nan,
    }
    if feedback is not None or local_heads is not None:
        raw = model.dfa_gradients(x_eval, y_eval, feedback) if feedback is not None else None
        local, _ = compute_gradients(model, x_eval, y_eval, method=method, feedback=feedback, local_heads=local_heads, args=args)
        diagnostics.update(conv_gradient_cosines(bp, local))
        if raw is not None:
            diagnostics["raw_projected_weight_step"] = conv_gradient_cosines(bp, raw)["projected_weight_step"]
            diagnostics["raw_hidden_projected_weight_step"] = conv_gradient_cosines(bp, raw)["hidden_projected_weight_step"]
            diagnostics["kndfa_weight_norm_ratio"] = conv_weight_norm_ratio(local, raw)
            diagnostics["kndfa_hidden_weight_norm_ratio"] = conv_hidden_weight_norm_ratio(local, raw)
    return {
        "dataset": args.dataset,
        "architecture": "manual_stride_convnet",
        "method": method,
        "feedback_mode": args.feedback_mode if feedback is not None else "none",
        "feedback_normalization": args.feedback_normalization if feedback is not None else "none",
        "optimizer": args.optimizer,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "gradient_clip": float(args.gradient_clip),
        "feedback_scale": float(args.feedback_scale),
        "seed": float(seed),
        "feedback_seed": float(feedback_seed),
        "feedback_rank": float(args.feedback_rank),
        "natural_damping": float(args.natural_damping),
        "n_train": float(args.n_train),
        "n_test": float(args.n_test),
        "label_noise": float(args.label_noise),
        "test_label_noise": float(args.test_label_noise),
        "epoch": float(epoch),
        "loss": bp.loss,
        "train_eval_acc": model.accuracy(x_eval, y_eval, batch_size=args.eval_batch_size),
        "test_acc": model.accuracy(x_test, y_test, batch_size=args.eval_batch_size),
        **diagnostics,
    }


def corrupt_labels(y: torch.Tensor, *, n_classes: int, noise: float, seed: int) -> torch.Tensor:
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


def apply_update(model: ManualConvNet, gradients, *, args: argparse.Namespace, state: dict[str, object]) -> None:
    if args.optimizer == "sgd":
        if args.weight_decay > 0:
            gradients = add_weight_decay(model, gradients, args.weight_decay)
        model.apply_gradients(gradients, lr=args.lr)
        return
    if args.optimizer != "adam":
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    state["step"] = int(state.get("step", 0)) + 1
    step = int(state["step"])
    beta1 = float(args.adam_beta1)
    beta2 = float(args.adam_beta2)
    eps = float(args.adam_eps)
    for name, get_param, set_param, grad in iter_parameter_grads(model, gradients):
        param = get_param()
        if args.weight_decay > 0:
            grad = grad + args.weight_decay * param
        m_key = f"{name}.m"
        v_key = f"{name}.v"
        if m_key not in state:
            state[m_key] = torch.zeros_like(param)
            state[v_key] = torch.zeros_like(param)
        m = state[m_key]
        v = state[v_key]
        assert isinstance(m, torch.Tensor)
        assert isinstance(v, torch.Tensor)
        m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        set_param(param - args.lr * m_hat / (torch.sqrt(v_hat) + eps))


def iter_parameter_grads(model: ManualConvNet, gradients):
    for idx in range(model.n_hidden_layers):
        yield (
            f"conv_weight.{idx}",
            lambda idx=idx: model.conv_weights[idx],
            lambda value, idx=idx: model.conv_weights.__setitem__(idx, value),
            gradients.conv_weights[idx],
        )
        yield (
            f"conv_bias.{idx}",
            lambda idx=idx: model.conv_biases[idx],
            lambda value, idx=idx: model.conv_biases.__setitem__(idx, value),
            gradients.conv_biases[idx],
        )
    for idx in range(len(model.fc_weights)):
        yield (
            f"fc_weight.{idx}",
            lambda idx=idx: model.fc_weights[idx],
            lambda value, idx=idx: model.fc_weights.__setitem__(idx, value),
            gradients.fc_weights[idx],
        )
        yield (
            f"fc_bias.{idx}",
            lambda idx=idx: model.fc_biases[idx],
            lambda value, idx=idx: model.fc_biases.__setitem__(idx, value),
            gradients.fc_biases[idx],
        )


def add_weight_decay(model: ManualConvNet, gradients, weight_decay: float):
    conv_weights = [g + weight_decay * w for g, w in zip(gradients.conv_weights, model.conv_weights)]
    conv_biases = [g + weight_decay * b for g, b in zip(gradients.conv_biases, model.conv_biases)]
    fc_weights = [g + weight_decay * w for g, w in zip(gradients.fc_weights, model.fc_weights)]
    fc_biases = [g + weight_decay * b for g, b in zip(gradients.fc_biases, model.fc_biases)]
    return type(gradients)(conv_weights, conv_biases, fc_weights, fc_biases, gradients.deltas, gradients.loss)


def clip_gradients(gradients, max_norm: float):
    parts = [g.flatten() for g in gradients.conv_weights] + [g.flatten() for g in gradients.conv_biases]
    parts.extend([g.flatten() for g in gradients.fc_weights])
    parts.extend([g.flatten() for g in gradients.fc_biases])
    total_norm = torch.linalg.vector_norm(torch.cat(parts)).clamp_min(1e-12)
    if float(total_norm.item()) <= max_norm:
        return gradients
    scale = max_norm / total_norm
    conv_weights = [g * scale for g in gradients.conv_weights]
    conv_biases = [g * scale for g in gradients.conv_biases]
    fc_weights = [g * scale for g in gradients.fc_weights]
    fc_biases = [g * scale for g in gradients.fc_biases]
    return type(gradients)(conv_weights, conv_biases, fc_weights, fc_biases, gradients.deltas, gradients.loss)


def final_summary(df: pd.DataFrame) -> pd.DataFrame:
    final = df.sort_values("epoch").groupby(["dataset", "method", "seed", "feedback_seed", "feedback_rank", "natural_damping"], as_index=False).tail(1)
    return (
        final.groupby(["dataset", "method"], as_index=False)
        .agg(
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", "sem"),
            train_mean=("train_eval_acc", "mean"),
            projected_step=("projected_weight_step", "mean"),
            hidden_projected_step=("hidden_projected_weight_step", "mean"),
            param_cosine=("param_cosine", "mean"),
            hidden_param_cosine=("hidden_param_cosine", "mean"),
            n=("test_acc", "size"),
        )
        .sort_values(["dataset", "test_mean"], ascending=[True, False])
    )


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for method, sub in df.groupby("method"):
        curve = sub.groupby("epoch")["test_acc"].mean()
        axes[0].plot(curve.index, curve.values, marker="o", label=method)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test accuracy")
    axes[0].legend(frameon=False, fontsize=7)
    final = df.sort_values("epoch").groupby(["method", "seed", "feedback_seed", "feedback_rank", "natural_damping"], as_index=False).tail(1)
    by_method = final.groupby("method")["test_acc"].mean().sort_values()
    axes[1].barh(by_method.index, by_method.values)
    axes[1].set_xlabel("Final test accuracy")
    step_col = "hidden_projected_weight_step" if "hidden_projected_weight_step" in df.columns else "projected_weight_step"
    for method, sub in df[df["method"] != "bp"].groupby("method"):
        axes[2].plot(sub.groupby("epoch")[step_col].mean(), marker="o", label=method)
    axes[2].axhline(0.0, color="0.6", linewidth=0.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Hidden projected BP step")
    fig.savefig(output_dir / "dfa_convnet_summary.png", dpi=220)
    plt.close(fig)


def write_report(df: pd.DataFrame, output_dir: Path) -> None:
    summary = final_summary(df)
    final = df.sort_values("epoch").groupby(["dataset", "method", "seed", "feedback_seed", "feedback_rank", "natural_damping"], as_index=False).tail(1)
    scores = predictor_scores(
        final,
        target="test_acc",
        predictors=[
            col
            for col in [
                "feedback_rank",
                "natural_damping",
                "projected_weight_step",
                "hidden_projected_weight_step",
                "param_cosine",
                "hidden_param_cosine",
                "activity_cosine_mean",
                "kndfa_weight_norm_ratio",
                "kndfa_hidden_weight_norm_ratio",
            ]
            if col in final.columns and pd.to_numeric(final[col], errors="coerce").notna().any()
        ],
    )
    write_markdown_report(
        output_dir / "dfa_convnet_report.md",
        title="Convolutional DFA/K-nDFA Baselines",
        sections=[
            (
                "Design",
                "Manual stride-conv network with explicit DFA feedback to each convolutional activation. nDFA preconditions conv filters by input-channel covariance; K-nDFA additionally preconditions by feedback-error channel covariance.",
            ),
            ("Final Summary", dataframe_to_markdown(summary, float_format=".4f")),
            ("Predictors", dataframe_to_markdown(scores, float_format=".4f")),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="results/dfa_convnet_baselines")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["mnist", "fashion_mnist", "cifar10", "cifar100", "imagenet100", "imagenet_subset", "imagenet", "imagenet1k"],
    )
    parser.add_argument("--data-dir", default="data/torchvision")
    parser.add_argument("--imagenet-root", default="")
    parser.add_argument("--imagenet-classes", type=int, default=0)
    parser.add_argument("--class-subset-seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--balanced-subsample", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["bp", "dfa_random", "ndfa_random", "ndfa_random_kronecker"])
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-feedback-seeds", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--label-noise", type=float, default=0.0, help="Fraction of training labels replaced by an incorrect class.")
    parser.add_argument("--test-label-noise", type=float, default=0.0, help="Optional test-label corruption; defaults to clean evaluation.")
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 2])
    parser.add_argument("--kernels", type=int, nargs="+", default=None)
    parser.add_argument("--paddings", type=int, nargs="+", default=None)
    parser.add_argument("--fc-hidden", type=int, nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--feedback-mode", choices=["flat", "channel"], default="flat")
    parser.add_argument("--feedback-normalization", choices=["none", "sqrt_target", "sqrt_flat"], default="none")
    parser.add_argument("--feedback-rank", type=int, default=0)
    parser.add_argument("--natural-damping", type=float, default=0.3)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Per-step wall-clock overhead of BP, DFA, nDFA, K-nDFA, and amortized-refresh nDFA.

Times one full training step (forward + local gradients + preconditioning +
parameter update) on the paper's actual MLP configurations: the synthetic-suite
MLP from ``run_dfa_multioutput_synthetic.py`` (input 64, hidden 256-128,
8 classes, batch 128) and the vision-MLP defaults from
``run_dfa_nmnc_comparison.py`` (flattened 28x28 input, hidden 512-256,
10 classes, batch 128). The ``ndfa_refresh_k*`` variants recompute the damped
covariance inverse every k batches (``--cov-refresh-interval`` in the runners,
via the same ``natural_precondition_gradients`` cache path) and reuse the
cached inverse otherwise. Data content is random; only timing is measured.
Writes one CSV row per (config, variant) and prints ratios relative to BP.

Usage:
    python experiments/benchmark_overhead.py --device cpu
    python experiments/benchmark_overhead.py --device cuda
    python experiments/benchmark_overhead.py --quick   # tiny CPU smoke test
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dfa_synthetic import natural_precondition_gradients  # noqa: E402
from infogeo.dfa import ManualMLP, init_feedback  # noqa: E402


# Paper MLP configs (see module docstring for provenance).
CONFIGS = {
    "synthetic_mlp": {"input_dim": 64, "hidden_dims": [256, 128], "n_classes": 8, "batch_size": 128},
    "vision_mlp": {"input_dim": 784, "hidden_dims": [512, 256], "n_classes": 10, "batch_size": 128},
    "tiny_smoke": {"input_dim": 16, "hidden_dims": [32, 16], "n_classes": 4, "batch_size": 32},
}
DEFAULT_CONFIGS = ["synthetic_mlp", "vision_mlp"]


def build_variants(refresh_intervals: list[int]) -> list[dict]:
    variants = [
        {"variant": "bp", "kind": "bp", "mode": "", "refresh_interval": 0},
        {"variant": "dfa", "kind": "dfa", "mode": "", "refresh_interval": 0},
        {"variant": "ndfa", "kind": "ndfa", "mode": "activity", "refresh_interval": 0},
        {"variant": "kndfa", "kind": "ndfa", "mode": "kronecker", "refresh_interval": 0},
    ]
    for k in refresh_intervals:
        variants.append(
            {"variant": f"ndfa_refresh_k{k}", "kind": "ndfa_refresh", "mode": "activity", "refresh_interval": int(k)}
        )
    return variants


def time_variant(
    *,
    config_name: str,
    config: dict,
    variant: dict,
    device: str,
    warmup: int,
    steps: int,
    damping: float,
    lr: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    model = ManualMLP(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        output_dim=config["n_classes"],
        seed=seed,
        device=device,
    )
    generator = torch.Generator().manual_seed(seed + 1)
    n_batches = 16
    xs = torch.randn(n_batches, config["batch_size"], config["input_dim"], generator=generator).to(device)
    ys = torch.randint(0, config["n_classes"], (n_batches, config["batch_size"]), generator=generator).to(device)

    kind = variant["kind"]
    feedback = None
    if kind != "bp":
        feedback = init_feedback(model, mode="random", seed=seed + 2, scale=1.0, rank=None)
    cache: dict | None = {} if kind == "ndfa_refresh" else None
    refresh_interval = max(int(variant["refresh_interval"]), 1)

    def one_step(step_idx: int) -> None:
        x = xs[step_idx % n_batches]
        y = ys[step_idx % n_batches]
        if kind == "bp":
            gradients = model.bp_gradients(x, y)
        else:
            gradients = model.dfa_gradients(x, y, feedback)
            if kind in {"ndfa", "ndfa_refresh"}:
                gradients = natural_precondition_gradients(
                    model,
                    gradients,
                    x,
                    damping=damping,
                    mode=variant["mode"],
                    cache=cache,
                    refresh=(step_idx % refresh_interval == 0) if cache is not None else True,
                )
        model.apply_gradients(gradients, lr=lr)

    is_cuda = torch.device(device).type == "cuda"
    for step_idx in range(warmup):
        one_step(step_idx)
    if is_cuda:
        torch.cuda.synchronize()
    times_ms = np.empty(steps)
    for step_idx in range(warmup, warmup + steps):
        start = time.perf_counter()
        one_step(step_idx)
        if is_cuda:
            torch.cuda.synchronize()
        times_ms[step_idx - warmup] = (time.perf_counter() - start) * 1e3

    return {
        "config": config_name,
        "variant": variant["variant"],
        "mode": variant["mode"],
        "refresh_interval": variant["refresh_interval"],
        "device": device,
        "device_name": torch.cuda.get_device_name(device) if is_cuda else (platform.processor() or "cpu"),
        "input_dim": config["input_dim"],
        "hidden_dims": "-".join(str(d) for d in config["hidden_dims"]),
        "n_classes": config["n_classes"],
        "batch_size": config["batch_size"],
        "warmup_steps": warmup,
        "timed_steps": steps,
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std(ddof=1)) if steps > 1 else float("nan"),
        "median_ms": float(np.median(times_ms)),
        "min_ms": float(times_ms.min()),
        "torch_version": torch.__version__,
    }


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")

    config_names = list(args.configs) if args.configs else list(DEFAULT_CONFIGS)
    if args.quick:
        config_names = ["tiny_smoke"]
        args.warmup = min(args.warmup, 2)
        args.steps = min(args.steps, 5)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = build_variants(args.refresh_intervals)
    rows = []
    for config_name in config_names:
        config = CONFIGS[config_name]
        bp_mean = None
        for variant in variants:
            row = time_variant(
                config_name=config_name,
                config=config,
                variant=variant,
                device=args.device,
                warmup=args.warmup,
                steps=args.steps,
                damping=args.damping,
                lr=args.lr,
                seed=args.seed,
            )
            if variant["variant"] == "bp":
                bp_mean = row["mean_ms"]
            row["ratio_vs_bp_mean"] = float(row["mean_ms"] / bp_mean) if bp_mean else float("nan")
            rows.append(row)
            print(
                f"{config_name:14s} {variant['variant']:16s} {args.device:5s} "
                f"mean={row['mean_ms']:8.3f}ms median={row['median_ms']:8.3f}ms "
                f"({row['ratio_vs_bp_mean']:.2f}x BP)",
                flush=True,
            )
    df = pd.DataFrame(rows)
    csv_path = output_dir / "benchmark_overhead.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/benchmark_overhead_v1")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--configs", nargs="+", default=None, choices=sorted(CONFIGS), help="Subset of configs to run; default synthetic_mlp + vision_mlp.")
    parser.add_argument("--refresh-intervals", type=int, nargs="+", default=[1, 10, 50])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--damping", type=float, default=0.3, help="Matches --natural-damping in the paper protocol.")
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Tiny CPU smoke configuration (<2 min).")
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Aggregate sharded DFA synthetic runs and regenerate analysis outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dfa_synthetic import plot_dfa_results, run_columns, write_dfa_analysis


def main() -> None:
    args = parse_args()
    shard_root = Path(args.shard_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for shard_dir in sorted(path for path in shard_root.glob("shard_*") if path.is_dir()):
        final_path = shard_dir / "dfa_synthetic_results.csv"
        partial_path = shard_dir / "dfa_synthetic_results.partial.csv"
        path = final_path if final_path.exists() else partial_path
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_shard"] = shard_dir.name
        frames.append(df)

    if not frames:
        raise SystemExit(f"No DFA synthetic shard CSVs found under {shard_root}")

    df = pd.concat(frames, ignore_index=True)
    dedup_cols = run_columns() + ["epoch", "layer"]
    df = df.drop_duplicates(subset=[col for col in dedup_cols if col in df.columns])
    csv_path = output_dir / "dfa_synthetic_results.csv"
    df.to_csv(csv_path, index=False)
    plot_dfa_results(df, output_dir)
    write_dfa_analysis(df, output_dir, early_epoch=args.early_epoch)

    n_runs = df[run_columns()].drop_duplicates().shape[0]
    print(f"Aggregated {len(frames)} shard files")
    print(f"Rows: {len(df):,}; unique runs: {n_runs:,}")
    print(f"Saved {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", default="results/dfa_synthetic_full_array/shards")
    parser.add_argument("--output-dir", default="results/dfa_synthetic_full_array")
    parser.add_argument("--early-epoch", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()

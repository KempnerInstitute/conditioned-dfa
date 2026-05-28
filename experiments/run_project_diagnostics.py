"""Run a compact Info-DFA project-level smoke diagnostic.

Fast harness for sanity-checking the core hidden-credit-assignment claims.
Not a replacement for the publication sweeps in ``run_dfa_*.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infogeo.analysis import dataframe_to_markdown, write_markdown_report
from infogeo.project_diagnostics import (
    InfoDfaDiagnosticConfig,
    run_infodfa_diagnostic,
    summarize_infodfa_diagnostic,
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = InfoDfaDiagnosticConfig(
        n_train=args.n_train,
        n_test=args.n_test,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        n_seeds=args.seeds,
        eval_size=args.eval_size,
        methods=tuple(args.methods),
    )

    dfa = run_infodfa_diagnostic(cfg)
    dfa.to_csv(output_dir / "infodfa_project_diagnostic.csv", index=False)

    summary = summarize_infodfa_diagnostic(dfa)
    write_markdown_report(
        output_dir / "project_diagnostics.md",
        title="Info-DFA Project Diagnostics",
        sections=[
            (
                "Purpose",
                "Fast smoke diagnostic for the Info-DFA hidden-credit-assignment claims "
                "on a known latent manifold. Tests BP, DFA, tangent-biased/orthogonal "
                "feedback, and activity/Kronecker nDFA.",
            ),
            ("Final By Method", dataframe_to_markdown(summary, float_format=".4f")),
        ],
    )

    print(f"Saved Info-DFA project diagnostic to {output_dir}")
    print("\nfinal_by_method:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/project_diagnostics")
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--eval-size", type=int, default=96)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bp", "dfa_random", "dfa_tangent_biased", "dfa_tangent_orthogonal", "ndfa_random", "ndfa_random_kronecker"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

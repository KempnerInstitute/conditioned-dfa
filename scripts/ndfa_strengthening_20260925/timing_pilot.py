"""Validation-only runtime pilot; not a selected scientific comparison."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments import run_ndfa_submission_benchmark as benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    data = None
    for method in ("bp", "dfa", "bp_activity", "dfa_activity", "bp_batchnorm", "dfa_batchnorm", "bp_decorrelation", "dfa_decorrelation"):
        run_args = benchmark.parse_args([
            "--output-dir", str(out / method), "--data-dir", args.data_dir,
            "--method", method, "--epochs", "2", "--lr", "0.001",
            "--feedback-scale", "0.1", "--activity-rho", "1",
            "--model-seed", "925001", "--feedback-seed", "925002",
            "--order-seed", "925003", "--augmentation-seed", "925004",
            "--split-seed", "925005", "--device", "cuda", "--threads", "4",
        ])
        benchmark.configure(run_args)
        if data is None:
            data = benchmark.load_data(run_args)
        _, result = benchmark.run(run_args, data=data)
        rows.append(result)
        (out / "pilot_summary.json").write_text(json.dumps({
            "purpose": "Runtime and implementation pilot only; common learning rate is not a tuned comparison.",
            "official_test_loaded": False, "results": rows,
        }, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

"""Reconstruct correction evidence without changing historical result files."""
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/ndfa_strengthening_20260925"
INPUTS = {}


def read(path, **kwargs):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, **kwargs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    legacy = ROOT.parent / "Info-Man/results"
    keys = ["condition", "n_train", "train_label_noise", "input_noise", "method", "seed", "feedback_seed", "feedback_rank"]
    d = read(legacy / "infodfa_multioutput_noise_sweep_aggregate_v2/dfa_multioutput_all.csv", usecols=keys + ["epoch", "loss", "test_acc"])
    f = d[d.feedback_rank.eq(0)].sort_values("epoch").groupby(keys).tail(1)
    f.to_csv(OUT / "synthetic_endpoints.csv", index=False)
    s = f.groupby(["condition", "method"]).agg(accuracy=("test_acc", "mean"), median_loss=("loss", "median"), runs=("loss", "size"), loss_above_10=("loss", lambda x: x.gt(10).mean()))
    s.to_csv(OUT / "synthetic_stability.csv")

    cohorts = []
    for name, cutoff in [("dfa_stall_threefactor", 55), ("dfa_stall_fashion_threefactor", 75), ("dfa_relu_mnist_threefactor", 108)]:
        d = read(ROOT / f"results/{name}_analysis_v1/confirmation_seed_means.csv")
        for cohort, sub in [("original", d[d.seed < cutoff]), ("extension", d[d.seed >= cutoff]), ("pooled", d)]:
            assert sub.n_feedback_seeds.eq(3).all()
            sub.to_csv(OUT / f"{name}_{cohort}_seed_means.csv", index=False)
            for method, g in sub.groupby("method"):
                cohorts.append(dict(study=name, cohort=cohort, method=method, n=len(g), seeds=" ".join(map(str, g.seed.tolist())), accuracy_pct=100*g.test_acc.mean(), accuracy_sem_pp=100*g.test_acc.sem(), loss=g.test_loss.mean(), loss_sem=g.test_loss.sem()))
    pd.DataFrame(cohorts).to_csv(OUT / "factor_cohort_summary.csv", index=False)

    # Adam configurations are intentionally preserved as the archived test-selected choices.
    best = read(ROOT / "results/infodfa_adam_diagk_aggregate_v1/infodfa_adam_diagk_best_endpoints.csv")
    raw = read(ROOT / "results/infodfa_adam_diagk_aggregate_v1/infodfa_adam_diagk_all.csv")
    adam = []
    adam_endpoints = []
    for _, b in best[best.method.isin(["dfa_sgd", "dfa_adam_hidden", "dfa_diag_activity_sqrt", "ndfa_activity"])].iterrows():
        g = raw[(raw.cell == b.cell) & (raw.method == b.method)]
        for col in ["feedback_rank", "damping", "adaptive_lr"]:
            g = g[g[col].isna()] if pd.isna(b[col]) else g[np.isclose(g[col], b[col])]
        g = g.sort_values("epoch").groupby(["seed", "feedback_seed"]).tail(1)
        seed = g.groupby("seed").test_acc.mean()
        assert len(seed) == 5 and len(g) == 10
        adam_endpoints.append(g)
        np.testing.assert_allclose(seed.mean(), b.test_mean, atol=1e-12)
        adam.append(dict(cell=b.cell, method=b.method, accuracy_pct=100*seed.mean(), sem_pp=100*seed.sem(), n_seeds=5, feedback_draws=2, damping=b.damping, adaptive_lr=b.adaptive_lr))
    pd.DataFrame(adam).to_csv(OUT / "adam_seed_summary.csv", index=False)
    pd.concat(adam_endpoints).to_csv(OUT / "adam_selected_endpoints.csv", index=False)

    cell = ["condition", "n_train", "train_label_noise", "input_noise"]
    frames = []
    for folder, method in [("infodfa_bn_baseline_v1", "dfa_random"), ("infodfa_bn_ndfa_synthetic_v1", "ndfa_random")]:
        files = sorted((ROOT / "results" / folder / "synthetic").glob("*/ntrain_*/label_*/input_*/dfa_multioutput_results.csv"))
        assert len(files) == 128
        d = pd.concat([read(p, usecols=keys + ["epoch", "test_acc"]) for p in files])
        d = d[(d.method == method) & d.feedback_rank.eq(0)].sort_values("epoch").groupby(keys).tail(1)
        assert len(d) == 1920 and d.epoch.eq(14).all()
        frames.append(d)
    pd.concat(frames).to_csv(OUT / "bn_endpoints.csv", index=False)
    w = pd.concat(frames).pivot(index=cell+["seed", "feedback_seed"], columns="method", values="test_acc")
    assert not w.isna().any().any()
    w["delta"] = 100*(w.ndfa_random - w.dfa_random)
    seed = w.groupby(level=["condition", "seed"]).mean()
    seed.to_csv(OUT / "bn_seed_means.csv")
    rows = []
    for condition, g in seed.groupby(level="condition"):
        interval = stats.t.interval(.95, 4, loc=g.delta.mean(), scale=g.delta.sem())
        rows.append(dict(condition=condition, dfa_bn_pct=100*g.dfa_random.mean(), a_bn_pct=100*g.ndfa_random.mean(), delta_pp=g.delta.mean(), sem_pp=g.delta.sem(), t95_low_pp=interval[0], t95_high_pp=interval[1]))
    pd.DataFrame(rows).to_csv(OUT / "bn_headtohead.csv", index=False)

    d = read(ROOT / "results/infodfa_seedlevel_stats_corrected_20260918/seedlevel_seed_deltas.csv")
    rows = []
    for condition, g in d.groupby("condition"):
        x = g.mean_delta_pp
        lo, hi = stats.t.interval(.95, 4, loc=x.mean(), scale=x.sem())
        rows.append(dict(condition=condition, mean_delta_pp=x.mean(), t95_low_pp=lo, t95_high_pp=hi))
    pd.DataFrame(rows).to_csv(OUT / "synthetic_t_intervals.csv", index=False)
    d.to_csv(OUT / "synthetic_seed_deltas.csv", index=False)

    vision = read(legacy / "infodfa_vision_noise_sweep_aggregate_v2/dfa_nmnc_summary.csv")
    assert vision.feedback_rank.eq(0).all()
    assert vision.groupby(["dataset", "n_train", "label_noise", "method"]).size().eq(1).all()
    vision.to_csv(OUT / "original_vision_summary.csv", index=False)
    vision.groupby(["dataset", "method"]).test_mean.mean().to_csv(OUT / "original_vision_means.csv")

    failure_records = []
    for folder in ["ndfa_submission_development_20260915", "ndfa_submission_development_v2_20260915", "ndfa_submission_development_recovery_20260915"]:
        for path in sorted((ROOT / "results" / folder).glob("shard_*/cases/*/endpoint.json")):
            item = json.loads(path.read_text())
            INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            failure_records.append(dict(root=folder, path=str(path.relative_to(ROOT)), record=item))
    (OUT / "development_endpoint_records.json").write_text(json.dumps(failure_records, indent=2)+"\n")

    chronology = []
    for folder in ["dfa_stall_kndfa_separate_damping_confirmation_v1", "dfa_stall_kndfa_separate_damping_dev_v1", "dfa_stall_threefactor_dev_v1/error_d10", "dfa_stall_threefactor_confirmation_v1/fb0"]:
        for path in sorted((ROOT / "results" / folder).glob("**/dfa_stall_comparison_summary.csv")):
            d = read(path)
            chronology.append(dict(path=str(path.relative_to(ROOT)), mtime_ns=path.stat().st_mtime_ns, seeds=d.seed.unique().tolist(), methods=d.method.unique().tolist(), columns=list(d.columns)))
    (OUT / "mnist_provenance.json").write_text(json.dumps({'caution':'Filesystem modification times are supporting evidence, not a complete execution history.', 'records':chronology}, indent=2)+"\n")
    (OUT / "input_sha256.json").write_text(json.dumps(INPUTS, indent=2)+"\n")
    print(f"Reconstructed correction evidence from {len(INPUTS)} inputs in {OUT}")


if __name__ == "__main__":
    main()

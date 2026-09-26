"""Independently check the frozen joint report after all test evaluations finish."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


def main(root, output=None):
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = root / 'MANIFEST.json'
    if manifest.exists():
        for name, record in read(manifest)['files'].items():
            assert sha(root / name) == record['sha256'], name
    decision = read(root / 'decision.json')
    freeze = read(root / 'joint_confirmation.json')
    plan = read(root / 'plan.json')
    gate = read(root / 'confirmation_training_gate.json')
    assert gate['passed'] and gate['joint_freeze_sha256'] == sha(root / 'joint_confirmation.json')
    expected = set(plan['confirmation_seeds'])
    data = {}
    tables = []
    run_count = 0
    for name, entry in freeze['configs'].items():
        # Original absolute paths are provenance. Always inspect the local
        # export, including when the original workspace still exists.
        config_path = root / name / 'config.json'
        assert sha(config_path) == entry['sha256']
        cfg = read(config_path)
        result = read(root / name / 'test_evaluation/summary.json')
        assert result['config_sha256'] == entry['sha256']
        expected_tasks = {(cfg['cases'][t['case_index']]['id'], t['seed']) for t in cfg['tasks']}
        actual_tasks = {(r['case']['id'], r['seed']) for r in result['rows']}
        assert actual_tasks == expected_tasks and len(result['rows']) == len(expected_tasks)
        assert all(r['status'] == 'complete' for r in result['rows'])
        run_count += len(result['rows'])
        groups = defaultdict(list)
        for row in result['rows']:
            groups[(row['case']['cell'], row['case']['family'])].append(row)
        which = 'final' if name in {'confirm_epochs', 'confirm_geometry', 'confirm_timing'} else 'best'
        for (cell, family), rows in groups.items():
            assert len(rows) == len(expected) and {r['seed'] for r in rows} == expected
            if any(r['status'] != 'complete' for r in rows):
                continue
            ordered = sorted(rows, key=lambda r: r['seed'])
            data[(name, cell, family)] = ordered
            tables.append(dict(study=name, cell=cell, family=family, checkpoint=which,
                               accuracy_percent=100 * np.mean([r['metrics'][which]['accuracy'] for r in ordered]),
                               loss=np.mean([r['metrics'][which]['loss'] for r in ordered]),
                               work_seconds=np.mean([r['training_seconds'] for r in ordered]),
                               interrupted_runs=sum(r['attempts'] != 1 for r in ordered)))

    def values(study, family, cell='cifar10', metric='accuracy', which='best'):
        rows = data.get((study, cell, family))
        if rows is None:
            return None
        cfg = read(root / study / 'config.json')
        if (cfg['budget_kind'] == 'work' or metric == 'work') and any(r['attempts'] != 1 for r in rows):
            return None
        if metric == 'direction':
            raw = [r.get('mean_early_direction_change') for r in rows]
        elif metric == 'work':
            raw = [r['training_seconds'] for r in rows]
        else:
            raw = [r['metrics'][which][metric] for r in rows]
        return None if any(v is None for v in raw) else np.asarray(raw, dtype=float)

    def contrast(study, left, right, cell='cifar10', metric='accuracy', which='best'):
        a = values(study, left, cell, metric, which)
        b = values(study, right, cell, metric, which)
        return None if a is None or b is None else (a - b) * (100 if metric == 'accuracy' else -1)

    primary = {}
    cf = 'confirm_foof_cifar'
    for family in ['activity', 'early_activity', 'foof_dfa']:
        primary['cifar_' + family] = contrast(cf, family, 'dfa_adamw')
    a = contrast(cf, 'foof_dfa', 'dfa_sgd')
    b = contrast(cf, 'foof_bp', 'bp_sgd')
    primary['FOOF_DFA_minus_BP_gain'] = None if a is None or b is None else a - b
    for family in ['full', 'centered']:
        primary['geometry_' + family] = contrast('confirm_geometry', family, 'diagonal_covariance_plus_mean', which='final')
    a = contrast('confirm_timing', 'dfa_0.0003_early', 'dfa_0.0003_late', which='final')
    b = contrast('confirm_timing', 'bp_0.0003_early', 'bp_0.0003_late', which='final')
    primary['timing_DFA'] = a
    primary['timing_DFA_minus_BP'] = None if a is None or b is None else a - b
    for other in ['activity', 'early_diagonal']:
        for metric in ['accuracy', 'loss']:
            primary[f'error_vs_{other}_{metric}'] = contrast('confirm_error_cifar', 'early_k', other, metric=metric)
    direction = values('confirm_error_cifar', 'early_k', metric='direction')
    primary['error_direction_above_threshold'] = None if direction is None else np.where(np.abs(direction - .001) < 1e-12, 0., direction - .001)
    for dataset in ['mnist', 'fashion']:
        a = contrast('confirm_benchmarks', 'activity', 'dfa_bn', dataset + '_input_nuisance')
        b = contrast('confirm_benchmarks', 'activity', 'dfa_bn', dataset + '_clean')
        primary[dataset + '_nuisance_accuracy'] = a
        primary[dataset + '_nuisance_loss'] = contrast('confirm_benchmarks', 'activity', 'dfa_bn', dataset + '_input_nuisance', metric='loss')
        primary[dataset + '_nuisance_minus_clean_gain'] = None if a is None or b is None else a - b
    for metric in ['accuracy', 'loss']:
        primary['standard_background_' + metric] = contrast('confirm_benchmarks', 'activity', 'dfa_bn', 'mnist_background_standard', metric=metric)
    assert len(primary) == 21 and set(primary) == set(decision['primary_tests'])
    raw_p = {}
    for key, vector in primary.items():
        row = decision['primary_tests'][key]
        if vector is None:
            assert row['mean'] is None
            raw_p[key] = 1.
            continue
        assert len(vector) == 10 and np.isfinite(vector).all()
        np.testing.assert_allclose(row['values'], vector, atol=1e-10, rtol=1e-9)
        avg, se = float(vector.mean()), float(stats.sem(vector))
        p = float(stats.ttest_1samp(vector, 0).pvalue) if se else (0. if avg else 1.)
        half = stats.t.ppf(.975, len(vector) - 1) * se
        np.testing.assert_allclose([row['mean'], row['low'], row['high'], row['p']],
                                   [avg, avg - half, avg + half, p], atol=1e-10, rtol=1e-8)
        raw_p[key] = p
    ordered = sorted(raw_p, key=raw_p.get)
    adjusted = np.minimum(1., np.maximum.accumulate([raw_p[k] * (21 - i) for i, k in enumerate(ordered)]))
    for key, p in zip(ordered, adjusted):
        np.testing.assert_allclose(decision['primary_tests'][key]['holm_p'], p, atol=1e-10, rtol=1e-8)
    efficiency = decision['temporary_activity_noninferiority']
    delta = contrast('confirm_epochs', 'early_bn', 'activity_bn', which='final')
    lower = float(delta.mean() - stats.t.ppf(.95, len(delta)-1) * stats.sem(delta))
    ratio = float(np.mean(values('confirm_epochs', 'early_bn', metric='work') /
                          values('confirm_epochs', 'activity_bn', metric='work')))
    np.testing.assert_allclose([efficiency['summary']['mean'], efficiency['lower_one_sided'], efficiency['mean_work_ratio']],
                               [delta.mean(), lower, ratio], atol=1e-10, rtol=1e-8)
    assert efficiency['passes'] == bool(lower > -.5 and ratio <= .8)
    error_pass = all(primary[k].mean() > 0 and decision['primary_tests'][k]['holm_p'] < plan['alpha']
                     for k in primary if k.startswith('error_'))
    assert error_pass == decision['error_factor_pass']
    report = dict(passed=True, primary_contrasts_verified=21, confirmation_runs_verified=run_count,
                                  efficiency_gate_verified=True, error_gate_verified=True,
                                  source='Independent reconstruction from held-out seed-level metrics',
                                  decision_sha256=sha(root / 'decision.json'),
                                  tables=tables,
                                  presentation_note='Fixed-epoch cohort is reported at final checkpoints here, matching its efficiency gate; the automatic decision.md overview uses best checkpoints for that cohort.')
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=True, primary_contrasts_verified=21, confirmation_runs_verified=run_count,
                         efficiency_gate_verified=True, error_gate_verified=True, output=str(output) if output else None)))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Optional audit output; archived evidence is never overwritten')
    args = parser.parse_args()
    main(args.root, args.output)

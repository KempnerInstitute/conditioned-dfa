"""Relocate a frozen confirmation cohort without changing its scientific settings.

This prepares files only. It never downloads data, trains, evaluates the test
set, or submits a scheduler job. Original evidence remains immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'assets/ndfa_confirmation_20260926'


def prepare(study, destination, data_dir, benchmark_root=None):
    original = EVIDENCE / study / 'config.json'
    config = json.loads(original.read_text())
    if config['dataset'] == 'benchmark' and benchmark_root is None:
        raise ValueError('Benchmark studies require --benchmark-root with a regenerated training-only cache')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = destination / 'source'
    shutil.copytree(EVIDENCE / 'source', source)
    for name, digest in config['source_sha256'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
    config.update(data_dir=str(Path(data_dir).resolve()), source_root=str(source),
                  output_root=str(destination / 'runs'))
    if benchmark_root is not None:
        cache = Path(benchmark_root).resolve()
        inventory = json.loads((cache / 'inventory.json').read_text())
        expected = config['benchmark_inventory']
        assert set(inventory) == set(expected)
        for name, item in inventory.items():
            # Serialized torch archives can differ while tensor content agrees.
            # Require the complete original data provenance, then pin the new bytes.
            assert item['provenance'] == expected[name]['provenance'], name
            assert hashlib.sha256((cache / item['file']).read_bytes()).hexdigest() == item['sha256'], name
        config.update(benchmark_root=str(cache), benchmark_inventory=inventory)
    path = destination / 'config.json'
    path.write_text(json.dumps(config, indent=2) + '\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = dict(study=study, original_config_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                   relocated_config_sha256=digest, tasks=len(config['tasks']),
                   changed_fields=['data_dir', 'source_root', 'output_root'] +
                       (['benchmark_root', 'benchmark_inventory'] if benchmark_root else []),
                   scientific_settings_and_seeds_unchanged=True, training_started=False)
    (destination / 'relocation_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(**receipt, config=str(path), runner=str(source / 'scripts/ndfa_strengthening_20260925/integrated_train.py')), indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, choices=sorted(p.parent.name for p in EVIDENCE.glob('confirm*/config.json')))
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--benchmark-root', type=Path)
    args = parser.parse_args()
    prepare(args.study, args.destination, args.data_dir, args.benchmark_root)

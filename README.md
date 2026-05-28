# Info-DFA: Conditioned Direct Feedback Alignment

Code, experiments, and figures for the ICLR 2026 paper *Conditioned Direct
Feedback Alignment for Noisy and Nuisance-Dominated Learning*.

This repository was split from `KempnerInstitute/Info-Geo` on 2026-05-28.
The companion **Info-Man** project (population geometry and behaviorally
used information) now lives at `KempnerInstitute/Info-Man`. The two
repositories share a small set of utilities (geometry, basic analysis helpers)
which are duplicated by design — see [`SPLIT_PLAN.md`](../Info-Geo/SPLIT_PLAN.md)
in the archived parent repo for the split rationale and the duplication policy.

## Layout

- `infogeo/`: reusable utilities — geometry, DFA / conv-DFA / NDFA training
  primitives, synthetic latent-manifold data, noise-correlation baselines,
  project-level diagnostics.
- `experiments/`: experiment drivers.
  - `run_dfa_synthetic.py`, `run_dfa_multioutput_synthetic.py`, `run_dfa_preconditioning_spectrum.py`:
    synthetic stress suite.
  - `run_dfa_vision_baselines.py`, `run_dfa_convnet_baselines.py`,
    `run_dfa_nmnc_comparison.py`, `run_dfa_coloredmnist.py`, `run_dfa_controls.py`:
    Fashion-MNIST / CIFAR / convnet / ColoredMNIST / reviewer-controls.
  - `run_imagenet_credit_assignment.py`, `evaluate_imagenet_torchvision_weights.py`:
    ImageNet-100 ResNet-18 diagnostics.
- `analysis/`: aggregators, paired-test scripts, table writers, paper-figure builder.
- `slurm/`: batch scripts for every experiment.
- `tests/`: pytest suite (15 tests).
- `drafts/Info-DFA/`: ICLR 2026 paper draft (nested git repo, pushes to
  `houman1359/Info-DFA-draft`).
- `RUNBOOK_iclr.md`: ICLR submission state, commands to reproduce every
  numerical claim, in-flight and completed cluster jobs.

## Quick start

```bash
python -m pytest                                            # 15 tests
python experiments/run_project_diagnostics.py --seeds 1     # ~30 sec smoke
python experiments/run_dfa_synthetic.py --quick             # ~1 min
python experiments/run_dfa_coloredmnist.py --n-seeds 1 --epochs 3   # ~10 sec
```

## Paper claims and where they come from

| Claim | Script |
|---|---|
| Synthetic 128-cell DFA-rescue (Table 4) | `run_dfa_multioutput_synthetic.py` + `aggregate_dfa_multioutput_synthetic.py` |
| Vision MLP noisy-label sweep (Table 5) | `run_dfa_nmnc_comparison.py` + `aggregate_dfa_nmnc_comparison.py` |
| Reviewer controls (Table 6) | `run_dfa_controls.py` + `write_infodfa_controls_table.py` |
| ColoredMNIST DFA-rescue (Table 7) | `run_dfa_coloredmnist.py` + `write_infodfa_coloredmnist_table.py` |
| Hard CIFAR-100 convnet (Table 8) | `run_dfa_convnet_baselines.py` |
| ImageNet-100 substitution-depth (Table 9) | `run_imagenet_credit_assignment.py` (slurm: `infodfa_imagenet_standard.sbatch`) |
| Paired statistical tests (Table 10) | `compute_infodfa_statistical_tests.py` + `write_infodfa_stat_tests_appendix.py` |

## Companion repo

The Info-Man population-geometry paper lives at
`KempnerInstitute/Info-Man` with code shared at the geometry-utility level
and otherwise independent. Both repos vendor copies of `infogeo/geometry.py`,
`infogeo/analysis.py`, and the project-shared parts of `infogeo/synthetic.py`.

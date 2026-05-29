# Info-DFA: Conditioned Direct Feedback Alignment

Code, experiments, and figures for the paper *Conditioned Direct Feedback
Alignment for Noisy and Nuisance-Dominated Learning*.

Direct Feedback Alignment (DFA) trains deep networks with fixed random feedback
instead of the transposed forward weights used by backpropagation (BP), but it
degrades sharply when the learning signal is dominated by noise, nuisance
variation, or limited samples. This project studies **covariance-conditioned
direct feedback** (nDFA / K-nDFA): feedback that is whitened against the
noise/nuisance covariance of the signal. The central finding is that
conditioning rescues DFA in noisy, low-sample, nuisance-dominated, and
mixed-context regimes — often matching or beating BP there — while remaining a
targeted improvement rather than a universal replacement for backpropagation.

## Layout

- `infogeo/`: reusable utilities — geometry, DFA / conv-DFA / nDFA training
  primitives, synthetic latent-manifold data, noise-correlation baselines, and
  project-level diagnostics.
- `experiments/`: experiment drivers.
  - `run_dfa_synthetic.py`, `run_dfa_multioutput_synthetic.py`,
    `run_dfa_preconditioning_spectrum.py`: synthetic stress suite.
  - `run_dfa_vision_baselines.py`, `run_dfa_convnet_baselines.py`,
    `run_dfa_nmnc_comparison.py`, `run_dfa_coloredmnist.py`,
    `run_dfa_controls.py`: Fashion-MNIST / CIFAR / convnet / ColoredMNIST /
    control studies.
  - `run_imagenet_credit_assignment.py`,
    `evaluate_imagenet_torchvision_weights.py`: ImageNet-100 ResNet-18
    diagnostics.
- `analysis/`: aggregators, paired-test scripts, table writers, and the
  paper-figure builder.
- `slurm/`: batch scripts for every experiment.
- `tests/`: pytest suite.
- `drafts/Info-DFA/`: paper draft (nested, separately versioned git repo that
  pushes to `houman1359/Info-DFA-draft`).

## Quick start

```bash
python -m pytest
python experiments/run_project_diagnostics.py --seeds 1          # ~30 sec smoke
python experiments/run_dfa_synthetic.py --quick                  # ~1 min
python experiments/run_dfa_coloredmnist.py --n-seeds 1 --epochs 3
```

## Main empirical claims

| Claim | Script |
|---|---|
| Synthetic 128-cell DFA rescue | `run_dfa_multioutput_synthetic.py` + `aggregate_dfa_multioutput_synthetic.py` |
| Vision MLP noisy-label sweep | `run_dfa_nmnc_comparison.py` + `aggregate_dfa_nmnc_comparison.py` |
| Control studies | `run_dfa_controls.py` + `write_infodfa_controls_table.py` |
| ColoredMNIST DFA rescue | `run_dfa_coloredmnist.py` + `write_infodfa_coloredmnist_table.py` |
| Hard CIFAR-100 convnet | `run_dfa_convnet_baselines.py` |
| ImageNet-100 substitution depth | `run_imagenet_credit_assignment.py` |
| Paired statistical tests | `compute_infodfa_statistical_tests.py` + `write_infodfa_stat_tests_appendix.py` |

## Companion repository

The Info-Man population-geometry project lives at `KempnerInstitute/Info-Man`.
The two projects are independent but share geometry and analysis helpers in
`infogeo/`.

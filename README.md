# Info-DFA: Conditioned Direct Feedback Alignment

Code, experiments, and figures for the paper *Conditioned Direct Feedback
Alignment via Activity and Error Geometry*.

Direct Feedback Alignment (DFA) trains deep networks with fixed random feedback
instead of the transposed forward weights used by backpropagation (BP), but it
degrades sharply when the learning signal is dominated by noise, nuisance
variation, or limited samples. This project studies a symmetric conditioned-DFA
family: **activity nDFA** right-preconditions the local update by a presynaptic
second moment, **error nDFA** left-preconditions by a local-error second moment,
and **K-nDFA** applies both factors. The fixed random feedback path is unchanged.
Activity conditioning has the broadest evidence in nuisance-stressed settings;
clean MNIST and preregistered Fashion-MNIST confirmations support the error
factor and a further two-sided gain, and both signs replicate on eight fresh
seeds in a ReLU/softmax MNIST model. BatchNorm remains a strong activity-side
alternative, vision rank sweeps are exploratory, and the separate ImageNet-100
block-output diagnostic is not the proposed weight-update operator.

## Scope and caveats

All error-side and two-sided results in the paper use per-example errors,
separate activity/error damping selected on validation data, layerwise norm
matching, and frozen multi-seed confirmations; earlier sweeps that formed the
error second moment from mean-loss-normalized deltas are excluded throughout.
The Fashion-MNIST study additionally compares local K-nDFA with a nonlocal
BP-error covariance source: the registered comparator reused the local damping
and was effectively activity nDFA after norm matching, so its equivalence
interpretation is withdrawn, and a post-hoc validation-retuned, fresh-seed
audit instead shows source specificity — the local DFA-error factor improves
activity nDFA, whereas the transported BP-error factor does not.

## Installation

Python 3.10+.

```bash
pip install -r requirements.txt
```

Core dependencies: torch 2.9, torchvision 0.24, timm 1.0, numpy, pandas,
scipy, matplotlib. MNIST, Fashion-MNIST, and CIFAR download automatically;
ImageNet-1k must be provided separately (see `REPRODUCE.md`).

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
  - `run_infodfa_adam_diagk_approx.py`: archived Adam/diagonal and two-sided
    approximation tests; only the activity-side comparisons support the paper.
    The decorrelation baseline (`dfa_actwhiten`, inverse-square-root
    preconditioning) runs through the multioutput synthetic driver.
  - `run_dfa_stall_comparison.py`: corrected activity/error/K-nDFA comparison
    with separate damping and train/validation/test separation.
  - `run_dfa_relu_vision_threefactor.py`: validation-safe ReLU/softmax
    architectural replication; `run_dfa_factorial_synthetic.py` contains the
    controlled activity/error intervention pilot.
  - `run_imagenet_credit_assignment.py`,
    `evaluate_imagenet_torchvision_weights.py`: ImageNet-100 ResNet-18
    diagnostics.
- `analysis/`: aggregators, paired-test scripts, table writers, and figure
  builders. The `analyze_dfa_stall_*.py` and
  `analyze_dfa_relu_vision_threefactor.py` scripts aggregate the tanh and ReLU
  factor confirmations and the post-hoc source-scale audit.
- `slurm/`: batch scripts for every experiment (site-specific headers; see
  `REPRODUCE.md`).
- `tests/`: pytest suite.
- `external/DFA-Stall/`: vendored reference implementation for the DFA-stall
  diagnostic (provenance in `external/DFA-Stall/VENDORED_INFO.md`).

## Quick start

```bash
python -m pytest
python experiments/run_project_diagnostics.py --seeds 1          # ~30 sec smoke
python experiments/run_dfa_synthetic.py --quick                  # ~1 min
python experiments/run_dfa_coloredmnist.py --n-seeds 1 --epochs 3
```

## Reproducing the paper

`REPRODUCE.md` maps every reported result to the script, parameter grid,
aggregator, and artifact root that regenerate it. `PREDICTIONS.md` and
`PREDICTIONS_SCORECARD.md` record the preregistered predictions and their
scored outcomes, including refuted ones.

| Claim | Script |
|---|---|
| Synthetic 128-cell DFA rescue | `run_dfa_multioutput_synthetic.py` + `aggregate_dfa_multioutput_synthetic.py` |
| Vision MLP noisy-label sweep | `run_dfa_nmnc_comparison.py` + `aggregate_dfa_nmnc_comparison.py` |
| Control studies | `run_dfa_controls.py` + `write_infodfa_controls_table.py` |
| ColoredMNIST DFA rescue | `run_dfa_coloredmnist.py` + `write_infodfa_coloredmnist_table.py` |
| Hard CIFAR-100 convnet | `run_dfa_convnet_baselines.py` |
| Activity/error/K-nDFA confirmations | `run_dfa_stall_comparison.py`, `run_dfa_relu_vision_threefactor.py`, and the corresponding three-factor analyses |
| ImageNet-100 substitution depth | `run_imagenet_credit_assignment.py` |
| Descriptive and seed-level sensitivity tests | `compute_infodfa_statistical_tests.py` + `compute_infodfa_seedlevel_stats.py` |

## Citation

```bibtex
@article{safaai2026conditioned,
  title   = {Conditioned Direct Feedback Alignment via Activity and Error Geometry},
  author  = {Safaai, Houman and Reddy, Varun and Sabatini, Bernardo L.},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

Released under the MIT License (see `LICENSE`).

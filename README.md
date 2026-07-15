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

Historical error-side and two-sided (`K-nDFA`) experiments used
mean-loss-normalized deltas when estimating the error second moment. Their
reported factor was therefore nearly scalar at the chosen damping and remains
excluded. The corrected DFA-stall experiments use per-example errors, separate
activity/error damping selected on validation data, norm matching, and frozen
five-seed by three-feedback-seed confirmations. Fashion-MNIST additionally
compares local K-nDFA with a nonlocal BP-error covariance source. The registered
source comparator reused the local damping and was effectively activity nDFA
after norm matching, so its original equivalence interpretation is withdrawn.
A post-hoc validation-retuned, fresh-seed audit instead shows source
specificity: the local DFA-error factor improves activity nDFA, whereas the
transported BP-error factor does not.

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
  - `analyze_dfa_stall_threefactor.py`,
    `analyze_dfa_stall_fashion_threefactor.py`,
    `analyze_dfa_stall_bpsource_retune.py`, and
    `analyze_dfa_relu_vision_threefactor.py`: tanh and ReLU factor
    confirmations plus the post-hoc source-scale audit.
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
| Activity/error/K-nDFA confirmations | `run_dfa_stall_comparison.py`, `run_dfa_relu_vision_threefactor.py`, and the corresponding three-factor analyses |
| ImageNet-100 substitution depth | `run_imagenet_credit_assignment.py` |
| Descriptive and seed-level sensitivity tests | `compute_infodfa_statistical_tests.py` + `compute_infodfa_seedlevel_stats.py` |

## Companion repository

The Info-Man population-geometry project lives at `KempnerInstitute/Info-Man`.
The two projects are independent but share geometry and analysis helpers in
`infogeo/`.

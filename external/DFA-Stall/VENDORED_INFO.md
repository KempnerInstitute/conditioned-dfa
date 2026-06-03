# Vendored Source

This directory vendors `git@github.com:varun04reddy/DFA-Stall.git` into the
Info-DFA repository for local DFA-stall diagnostics.

- Upstream commit imported: `3d88734`
- Imported on: 2026-06-03
- Local generated data, logs, caches, and bytecode are ignored by the parent
  repository.

## Info-DFA nDFA Stall Runs

The recent nDFA/K-nDFA stall diagnostics were not produced by modifying the
vendored upstream `train.py`. They were produced from the parent Info-DFA
repository with:

- `experiments/run_dfa_stall_comparison.py`
- `analysis/aggregate_dfa_stall_comparison.py`
- `analysis/aggregate_dfa_stall_error_ablation.py`

The runner imports this vendored DFA-Stall implementation to preserve the
original setup: MNIST, a 3-hidden-layer 300-unit tanh MLP, sigmoid outputs,
binary log loss, SGD with learning rate `1e-3`, batch size `128`, and fixed
random direct feedback. It then applies Info-DFA variants to the hidden-layer
DFA gradients:

- `dfa`: raw DFA hidden updates.
- `ndfa`: activity/input-side second-moment preconditioning.
- `endfa`: error/local-delta-side second-moment preconditioning only.
- `kndfa`: both activity-side and error-side preconditioning.

The main comparison used three seeds (`42 43 44`), 1000 training steps, hidden
width 300, probe set size 1024, and damping `0.3`, with and without
`--norm-match-hidden`. The norm-matched condition rescales each preconditioned
hidden weight gradient to the corresponding raw-DFA layerwise Frobenius norm;
it was used as a diagnostic to separate update direction from update size.

The error-side ablation swept damping values `{0.03, 0.1, 0.3, 1, 3, 10}` for
`dfa`, `ndfa`, `endfa`, and `kndfa`, again with and without
`--norm-match-hidden`.

Generated outputs are intentionally kept under the parent repository's ignored
`results/` directory:

- `results/dfa_stall_comparison_3seed_v1`
- `results/dfa_stall_comparison_normmatch_3seed_v1`
- `results/dfa_stall_comparison_overview_v1`
- `results/dfa_stall_error_ablation_damping_v1`

The exact shell commands are recorded in the parent `REPRODUCE.md` under
`External DFA-Stall diagnostic`.

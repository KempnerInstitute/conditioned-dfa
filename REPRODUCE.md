# Reproducing the September 26 confirmation revision

Use the current `main` branch or the matching `ndfa-confirmation-2026-09-26`
tag in the code and manuscript repositories. Earlier reproduction instructions
are preserved in [the historical guide](docs/research/reproduction_before_confirmation_20260926.md).

## Recompute reported statistics without training

```bash
python -m pip install -r requirements.txt
python scripts/ndfa_strengthening_20260925/audit_joint_decision.py \
  --root assets/ndfa_confirmation_20260926 \
  --output build/confirmation_verification.json
```

The audit checks all exported SHA256 values and reconstructs all 21 primary
contrasts from the eight original test-summary files. It verifies every
declared case/seed, paired means, individual 95% t intervals, p values, global
Holm adjustment, the five-part error gate, and the final-checkpoint efficiency
criterion. It uses final checkpoints for the geometry, constant-rate timing,
and fixed-epoch efficiency cohorts, and validation-CE-selected checkpoints for
work-budget cohorts. The original generated `decision.md` used best checkpoints
in its fixed-epoch overview; the audit and current manuscript use final ones.

This is an audit of saved seed-level metrics, not new checkpoint inference or
an independent measurement of hardware time. The endpoint records retain
checkpoint and prediction hashes; the large tensors are not included in Git.

## CPU implementation checks

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_integrated_round.py tests/test_foof.py \
  tests/test_local_preconditioning.py tests/test_activity_geometry.py \
  tests/test_decorrelated_dfa.py tests/test_decision_round.py \
  tests/test_ndfa_next_round.py tests/test_development_selection.py \
  tests/test_strengthening_controls.py tests/test_round2_design.py
```

These cover dense/sample-space operators, FOOF, forward decorrelation, frozen
selection, resumption, shared-checkpoint branches, error direction probes,
test-access gates, and bounded study designs. They do not launch GPU jobs.

## Build the manuscript and arXiv source bundle

```bash
git clone --branch ndfa-confirmation-2026-09-26 \
  https://github.com/houman1359/Info-DFA-draft.git drafts/Info-DFA
python drafts/Info-DFA/scripts/build_manuscript.py
python drafts/Info-DFA/scripts/build_arxiv_package.py \
  --output-dir drafts/Info-DFA/build/arxiv_upload \
  --reference-pdf drafts/Info-DFA/build/checked/conditioned_dfa_arxiv.pdf
```

pdfLaTeX and the LaTeX packages in the manuscript preamble are required. The
checked build verifies source/style pins, references, figure placement, author
order, and the nine-page ICLR main-text limit. The source bundle is rebuilt
after extraction outside the workspace and compared with the reviewed PDF.
The arXiv build omits the AI statement at the authors' request; the ICLR build
retains it. No upload is performed by these commands.

To regenerate the main confirmation table, appendix, and Figure 5, run
`python drafts/Info-DFA/scripts/confirmation_material.py`. That renderer uses
the manuscript's curated compact records. The independent audit above instead
uses the original test summaries, including the raw direction diagnostic.

## Prepare a training rerun

Obtain the relevant datasets in torchvision format. The original runs used
PyTorch 2.9, float32 training, float64 evaluation loss, and disabled TF32.
The paper records hardware separately: H100 for the main practical cohort,
H200 for several interventions, and RTX PRO 6000 Blackwell for the amended
background cohort. A different accelerator can change the number of updates
within a time budget, so wall-clock values are hardware dependent.

The helper copies the exact 46-file frozen source snapshot, preserves the
scientific settings and seeds, and rebases data/source/output paths:

```bash
python scripts/ndfa_confirmation_20260926/prepare_rerun.py \
  --study confirm_foof_cifar --destination /path/to/new_rerun \
  --data-dir /path/to/torchvision
```

It prints the relocated configuration path, hash, and training script. On an
allocated GPU, pass them to the runner:

```bash
python /path/to/new_rerun/source/scripts/ndfa_strengthening_20260925/integrated_train.py \
  --config /path/to/new_rerun/config.json \
  --config-sha256 HASH_FROM_RELOCATION_RECEIPT --task-index 0
```

Each task index identifies a frozen method/seed combination. Training accesses
only training/validation data and writes best/final checkpoints. This helper
does not schedule jobs, alter the original confirmation, or invoke official
test evaluation. The original full-study controller and gate are preserved in
the source snapshot for inspecting the complete experiment protocol.

For the seven background conditions, build the training-only cache with
`scripts/ndfa_strengthening_20260925/prepare_integrated_benchmarks.py` using
`--root`, `--data-dir`, and `--archive` (the original Larochelle
MNIST-background-images ZIP). Then supply `--benchmark-root` to the rerun
helper. It requires the original tensor/provenance hashes and pins the new
serialized cache bytes. Dataset archives and caches are not redistributed.

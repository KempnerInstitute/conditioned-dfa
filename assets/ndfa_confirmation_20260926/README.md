# Joint confirmation evidence

This export contains all 880 retained runs and the 21 frozen primary tests.
`MANIFEST.json` records the SHA256 and size of every exported scientific file;
those files are byte-identical to their archived sources. Do not edit them
to rebase paths or change a reported result.

- `plan.json`, `designs.json`, `protocol.md`, and `joint_confirmation.json`
  record development choices, the joint freeze, seeds, and hardware amendments.
- Each `confirm_*/config.json` identifies its cases and tasks. Its
  `test_evaluation/summary.json` contains every retained endpoint, checkpoint
  hashes, measured work, FLOP estimates, and available direction diagnostics.
- `source/` contains the exact 46-file source snapshot pinned by the protocol.
  The public top-level runner may have later scheduler maintenance; use the
  snapshot for retraining the recorded configuration.
- `claims/` preserves the interpretation map and timestamp before joint test
  evaluation. It was written after development and is not a preregistration
  preceding development.
- `scheduler/` preserves the five-run replay protocol and its verification.
  `reviews/` includes hardware accounting and the secondary synthetic result.
- `manuscript_compact_records.json` is the separately curated manuscript input.
  Its compact records omit some raw diagnostic fields; the original summaries
  above retain the error-direction measurements needed by the independent audit.

All work-budget tests require uninterrupted retained training runs. The five
scheduler-interrupted RTX cases were replayed once before test access under
a metadata-only rule. The entire background cohort changed from H200
development to RTX PRO 6000 Blackwell confirmation while retaining the
60-second budget. These deviations remain part of the reported protocol.

`decision.md` is an unchanged historical automatic report. Its fixed-epoch
overview uses best checkpoints; the corresponding efficiency gate, current
manuscript, and independent audit use final checkpoints. The analysis supports
activity gains, centered-covariance controls, and a bounded efficiency result;
the additional error-factor gate and background predictions do not pass.

Run the audit from the repository root as described in `REPRODUCE.md`.
Original absolute paths are provenance only. The auditor reads this directory;
`prepare_rerun.py` creates a separate relocated configuration for new training.
No datasets, checkpoints, prediction tensors, or new training results are
contained in this export.

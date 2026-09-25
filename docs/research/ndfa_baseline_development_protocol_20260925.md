# E1a: stable, long-horizon baseline development

This is the first experimental stage of the approved strengthening plan. It is validation development, not confirmation and not evidence of a new result in the manuscript.

The CIFAR-10 MLP retains widths 1024–512, fixed random feedback, and the exact current-batch activity operator. The official training set is split into 45,000 training and 5,000 validation images. Scaling uses training images only. The official test set is never loaded. Initialization, feedback, data order and augmentation have separate deterministic streams, paired across methods.

Nine method families receive twelve candidate settings each and two development seeds per setting: BP, DFA, activity-conditioned BP and activity-conditioned DFA, each with and without BN, plus DFA with forward decorrelation. Each family considers SGD with momentum and AdamW. Raw rules allocate their settings to learning rates; conditioned rules also vary relative damping; forward decorrelation also varies its adaptation rate. Equal trial counts do not imply equal search completeness. Boundary choices require a documented validation-only extension before confirmation.

All trials run for 200 epochs with five warmup epochs and a cosine schedule ending at 1% of peak rate. Hidden activity conditioning matches the norm of its own raw gradient; it does not use BP norms. Classifier, bias and BN affine updates are unchanged. Weight decay applies to weight matrices only. The feedback scale is fixed at 0.1 across local rules. These choices define this stage; later mechanism experiments must not conflate changing them with changing the operator.

Selection minimizes final validation cross-entropy averaged across both development seeds, then maximizes validation accuracy and finally uses case ID to break ties. A numerical failure or incomplete run makes a candidate ineligible. All failures remain in the results. Intermediate checkpoints describe learning curves and do not select models. A 200-epoch horizon is not assumed to establish convergence: inspect late training loss, validation loss and accuracy before deciding whether all competitive methods need longer development runs.

Training work includes sampling, augmentation, manual gradients, conditioning, optimizer steps and numerical checks. It excludes validation, evidence hashing and export. Curves provide descriptive accuracy–work comparisons; the final equal-work experiment will freeze its own budget, schedule and settings. Source hashes, configuration hashes, accelerator identity, precision, states and paired-stream fingerprints are saved. Training uses float32, deterministic algorithms and disabled TF32 on `kempner_h100_priority`.

The eight-method runtime pilot completed on one H100. A complete activity-plus-BN production case is run before launching the rest of the array. Concurrency is capped at four H100s. The array contains 108 configurations and 216 training runs. Measured runtimes will determine scheduling; this is not a request to consume an unbounded resource budget.

Required follow-up before a scientific conclusion:

- Add and audit a faithful EMA/amortized FOOF-BP implementation. The current BP+activity arm is an operator control, not the full FOOF optimizer.
- Run the separately tuned synthetic nuisance/low-sample/clean controls and stronger factor-study baselines.
- Resolve boundary selections, check convergence and freeze confirmation settings and replication counts before test evaluation.
- Compare matched work and final endpoints with paired uncertainty. Only then choose the centering and alignment interventions and the larger application.

Frozen inputs and job receipts are in `results/ndfa_strengthening_20260925/baseline_development_v1/`. The progress ledger records completion separately from launch; no new-development result has been inserted into the paper.

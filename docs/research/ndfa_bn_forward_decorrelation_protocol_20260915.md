# Prospective matched forward-decorrelation comparison — 15 September 2026

This study compares activity-conditioned DFA (A) with an actual forward-decorrelation mechanism under the successful CIFAR-10 BatchNorm protocol. BP and unconditioned DFA provide fresh context. The primary question is whether A improves validation accuracy over FD-DFA under the same measured update-work and finite development-search budgets. No outcome is available when this protocol is frozen. This is one fixed architecture and training family; it cannot establish globally tuned convergence or optimizer-independent superiority.

## Mechanism and reproduction boundary

The reference is Ahmad, [*Correlations Are Ruining Your Gradient Descent*, Section 3 and Algorithm 1](https://arxiv.org/html/2407.10780v2#S3), with [official dense implementation at revision `00cf47050bbd20e6a153e10bd86e1651524f9779`](https://github.com/nasiryahm/CorrelationsRuinGD/blob/00cf47050bbd20e6a153e10bd86e1651524f9779/crgd/decor.py). The pinned file is preserved with this study and compared directly in a CPU fixture.

Before every linear map, including input and classifier, FD computes `U=(H−mean(H))D` during training. It maintains an identity-initialized dense matrix and a zero-initialized running mean. The code's update is `D_next=gD−eta C(gD)`, with `C=U_SᵀU_S/k`, the first `k=floor(.1 n)+1` rows, and the mean original-to-transformed sample-norm ratio for `g`. Running-mean momentum is .1. Our unchanged primitive associates `C(gD)=U_Sᵀ(U_S(gD))/k`, avoiding a cubic matrix product. It retains the pinned code's multiplication orientation. Inference uses running means and frozen matrices.

This is explicitly **Ahmad's forward mechanism adapted to our direct feedback, ReLU, BN and SGD protocol**. The published FA/Adam experiments are not reproduced. See the earlier [source audit and convention discussion](ndfa_forward_decorrelation_baseline_2026-09-14.md).

Our fixed matrix `B_i` injects output error into the original post-ReLU hidden activity **before its outgoing decorator**. The local hidden delta therefore applies the existing ReLU and BN derivatives, while its weight gradient uses the decorated presynaptic input. DFA replaces the outgoing credit path as a whole; multiplying this signal by the outgoing `Dᵀ` would change its declared injection coordinates. A separate BP implementation, used only for CPU verification, composes the outgoing old-matrix/centering pullback with ReLU and BN. Tests compare it to full autograd and compare DFA to a local surrogate with the specified fixed injection site.

Decorator state updates exactly once per training forward, and never during validation or export. FD-DFA uses its raw local gradient without A's norm matching or a second conditioning forward. The final classifier uses decorated presynaptic activity. All matrix updates, centering, gain calculation, matrix applications, and primitive numerical guards are charged to FD's work clock. The matrices remain necessary inference state; they are not counted as supervised weight parameters.

## Fixed design and search

The four arms are BP, DFA, A and FD-DFA. All use the same 48,000 CIFAR-10 training identities, fixed 2,000 validation identities (split seed 80423), training-only channel standardization, 3072→1024→512→10 ReLU MLP, hidden affine BN, fixed DFA feedback scale .1, and 64 sampled identities with two independent crop/flip views per update. The supervised parameter count is 3,679,754. Model initialization, fixed-feedback realization, sample order and augmented-view prefixes are paired by seed. Different update counts consume different lengths of these same streams. Source and training-file hashes are frozen.

All arms use the previously reviewed SGD implementation: momentum .9, matrix-only weight decay .0005, no Nesterov, no dampening, no fused or foreach path. The common schedule warms up from .1 of peak to peak over the first 5% of measured work, then follows cosine decay to .01 of peak. For A, each instantaneous hidden gradient is norm-matched before common momentum and weight decay. The momentum-driven parameter displacement is not itself norm-matched.

Each arm gets **nine candidates × two development seeds**, hence 18 trials and 540 seconds of measured update work. BP/DFA spend these trials on a broader LR grid; A/FD-DFA spend them on a 3×3 method-specific grid. Equal trial budgets do not imply equal coverage of all hyperparameters.

| Arm | Peak learning rate | Method-specific setting |
|---|---|---|
| BP, DFA | .003, .01, .03, .06, .1, .2, .3, .6, 1 | none |
| A | .03, .1, .3 | relative activity damping 3, 30, 300 |
| FD-DFA | .03, .1, .3 | decorator learning rate 1e−6, 1e−5, 1e−4 |

The A/FD peak grid retains the prior .1 upper-bound winner and includes a higher value. FD's grid retains the pinned default 1e−5. Mean momentum, sample fraction, BN, architecture, augmentation, weight decay and optimizer family are not searched. The two development seeds are 38000–38001; confirmation uses eight fresh paired seeds 39000–39007. One timing probe per arm uses seed 38999 and candidate c04 for exactly 500 updates. There are **4 timing +72 development +32 confirmation models**, fixed before any outcomes.

Selection requires every declared development outcome. A candidate is eligible only when both runs finish with finite outcomes. Choose minimum final mean validation cross-entropy; ties use higher mean accuracy, then smaller peak LR, relative damping and decorator LR. Report boundary selections for every searched dimension. Do not extend the grid, substitute a seed, or discard a failed cell. The immutable selection artifact binds all 72 development cases before any confirmation starts. Failure of an entire method's candidate set stops the workflow without a confirmation claim.

## Measured work, diagnostics and failures

Every scientific case receives 30 seconds of synchronized update work using the reviewed carried clock. Sampling, augmentation, gradient construction, conditioning/decorrelation, SGD and recurring loop bookkeeping are charged. Validation, common full-state diagnostics, export, startup and data loading are excluded and reported separately. The last full update is completed and its overshoot is charged. This matches measured update work, not end-to-end wall time, kernel-only execution, FLOPs, or energy.

Record validation at initialization, every three seconds of work, the final endpoint, and step 5000 if reached. Finite checks include parameters, BN running state and decorator state; inherited primitive checks are also retained and charged inside FD updates. Numerical failure retains observed finite history and available model/optimizer/sampling diagnostics without labeling them a completed endpoint. CUDA, file, or unrelated runtime errors remain infrastructure failures. No retry or scientific fallback is permitted.

The parent never initializes CUDA. It reads the actual Slurm end time and launches isolated sequential children. After four fixed timing probes, a prospective gate projects all 104 scientific cases from the worst measured child overhead, adds missing validation observations, multiplies overhead by 1.1, and reserves 480 seconds for CPU audit/report/shutdown. It rechecks whether the entire remaining cohort fits before every case. A gate stop preserves all evidence and is not a negative scientific result. No partial confirmation subset supports an inference.

## Artifacts and analysis

Every successful model saves its final supervised/BN state, all decorator matrices and running means when present, decorator update counts, preprocessing tensors and their hashes, validation identities/labels/logits, complete declared history, and optimizer/sampling state. New checkpoint schema 2 is explicit. The frozen `exports.restore_model` and `exports.inference_logits` helpers support later separately declared inference from uint8 NCHW images, preserving learned state and RNG. These helpers load no dataset. Step5000 saves metrics only; exact training resume is not claimed.

The CPU audit checks exact case inventory, frozen source/config, selection timing and evidence, candidate settings, complete histories and triggers, reconstructed learning rates, work overshoot, initialization/preprocessing/validation parity, state shapes/finiteness, decorator counts, and independent float64 metrics from saved logits. Reports verify the accepted audit's bound case-artifact hashes. CPU fixtures also exercise state restoration and forward replay; full scientific-model GPU replay is not claimed by this validation audit.

The one primary comparison is **A minus FD-DFA final validation accuracy at 30 seconds**, with all eight paired values, their mean/sample SD, and paired t95% interval. A two-sided exact paired sign-flip test is supplemental and assumes exchangeable paired signs. CE, A−DFA, A−BP, FD-DFA−DFA, update count, peak memory, state size, measured update work and child duration are descriptive and always retained. Secondary step5000 contrasts require all eight pairs in the relevant two arms to reach that step; report missing endpoints rather than extrapolating. There is no pooling with older seeds or selection based on confirmation outcomes. Intervals describe training-seed variability conditional on the fixed, repeatedly used data/settings; they do not quantify new-dataset uncertainty.

## Resource and scope lock

Requested allocation: one H100 on `kempner_h100_priority`, account `kempner_dev`, QOS `kemp_gpu16_id38`, four CPUs, 24 GB host memory, **110 minutes**, no requeue, excluding `holygpu8a05402`. No job is submitted by preparation. Root review of the final source/config/packet is required before the single launch.

Prior phase actual is 5.7363895556 GPU-hours. The prospective 1.8333333333-hour study reservation plus root's separate .3333333333-hour final-evaluation reservation yields **7.9030562222/8 GPU-hours**. Actual terminal allocation replaces the study reservation, including any gate-stop allocation. No training child accesses official test data. Official test inference, including the acknowledgement of prior test use, belongs to root's separate locked protocol.

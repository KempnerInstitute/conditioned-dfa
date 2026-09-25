# nDFA: integrated plan for a stronger paper

**Decision, 25 September 2026:** prioritize a stronger paper over several weeks rather than optimize for the current deadline. This plan combines the two September 25 reviews, the earlier manuscript/equation review, the previous numerical audit, and the user's writing and figure preferences. It is a research plan, not a report that the proposed changes or experiments have been completed.

The current manuscript is reproducible but has unresolved attribution, baseline, and provenance problems. Correcting those is necessary. A stronger research contribution requires an explanation and a reliable benefit that remain after stable, individually tuned baselines. More seeds, more pages, or a larger architecture alone will not provide that contribution.

## 1. Scientific objective and decisions

The primary question should become:

> **When does activity conditioning improve the acquisition and use of approximate credit signals, beyond the improvements available from ordinary optimization and normalization?**

There are three separable outcomes to test: preventing unstable updates, shortening an unproductive alignment phase, and improving validation-selected generalization. A speed benefit is legitimate even if final accuracies converge, but it must be measured against tuned alternatives at equal work. A generic optimization benefit must not be presented as a DFA-specific mechanism.

The most promising route is an activity-first paper with a causal study of alignment and activity geometry. Keep error conditioning as a secondary result until it shows an interpretable benefit beyond scalar damping on stronger tasks. Treat unknown-noise suppression and a new learning rule as conditional research directions, not established properties.

**Decisions after the first baseline study:**

| Observation | Resulting direction |
|---|---|
| Activity conditioning improves stable DFA at useful work budgets and changes raw credit alignment under controlled interventions | Develop the DFA-specific mechanism as the main contribution. |
| It improves early compute efficiency but tuned endpoints agree | Write an optimization-efficiency paper with proper cost accounting and a narrower generalization claim. |
| Its benefits are reproduced by centering, BN, diagonal adaptation, or a standard optimizer | Explain that mechanism and evaluate the simpler solution; do not keep a stronger full-moment claim. |
| It improves BP and DFA similarly, without a distinctive alignment effect | Position it as a study of applying established conditioning to approximate credit, with limited algorithmic novelty. |
| No robust benefit survives fair tuning and controls | Stop the scale expansion; complete a rigorous narrower study and reconsider venue. |

Mock reviews identify likely objections, but their scores do not calibrate an acceptance probability. Neither 10% nor 20% is an evidence-based estimate for this paper. The current concerns are substantial enough that another cosmetic revision would be insufficient.

## 2. What has been checked, and what remains to verify

This planning pass checked primary literature, the active LaTeX sources, selected configurations, the old vision summaries, the original synthetic endpoint archive, and the review's saved BP reruns. The preceding [September 25 audit](ndfa_comment_audit_20260925.md) recomputed the BN contrasts, extra-seed contrasts, five-seed intervals, and validation-selected vision summaries. It did not establish the entire experimental chronology.

| Finding | Evidence/status | Required action |
|---|---|---|
| Activity operator overlaps FOOF | Confirmed against Benzing's Eq. 6; implementation details still differ | Explicit algebraic attribution and a faithful FOOF-BP baseline. |
| Main theoretical intuition has close precedents | Confirmed in the Amari and Heckel–Yilmaz sources | Separate established spectral effects from any new DFA-specific result. |
| Primary synthetic DFA has extreme finite losses | Recomputed: stressed-regime medians approximately 1,195–86,171; 97–100% of DFA endpoints exceed loss 10, versus none for activity nDFA | Describe high-loss instability and show loss alongside accuracy. These finite endpoints alone do not prove mathematical divergence. |
| Primary sweep's own BP differs from the displayed tuned reference | Recomputed stressed-regime BP accuracies 27.93/51.43/38.02%; task-aligned 91.99% | Identify each BP cohort and selection rule; use like-for-like primary comparisons. |
| MNIST confirmation seeds/test were exposed before validation development | Reported with specific times, not independently reconstructed in this pass | Highest-priority chronology audit; correct all affected claims, including caption and appendix language. |
| Extra model seeds support both error-side contrasts | Recomputed; original/additional/pooled cohorts are distinct | Report post-hoc replication without retroactive preregistration or test-blindness claims. |
| DFA+BN versus activity-nDFA+BN comparison is missing | Recomputed from both raw archives | Extend the existing BN table using global-seed uncertainty. |
| Original vision table supposedly selected feedback rank on test data | **Incorrect caveat:** original archive has rank 0 only and one configuration per method/cell | Correct the manuscript and the earlier audit's assumption. Audit the “Best NC” choice separately. |
| Validation-selected vision replication is complete | Reproduced all 192 selected records | Useful corroboration, but it trains on 80% of the old nominal sample sizes; describe it as a different cohort. |
| Longer work budgets use different seed sets | **Incorrect:** both use 61000–61004 in the configuration | State shared seeds, separate runs, and stretched schedules. They are not checkpoints of one trajectory. |
| Failure counts and three rounding differences | Specific claims in the review; not independently replayed here | Regenerate from classified run records, not by hand-editing reported numbers. |
| Original analysis summaries now pool extensions | Confirmed for the saved seed means | Export explicit original/extension/pooled products and bind every table/figure to a cohort manifest. |
| Anonymity issue and 27-entry bibliography file | Specific location/file not recovered in this pass | Locate the reported issue and bibliography; verify exact release contents and each cited claim. Do not infer an anonymity defect merely from arXiv being public. |

The reviewer’s CPU reruns support numerical portability for the tested settings; they do not establish hardware-independent throughput or identical behavior for every experiment. Preserve `logs/claude_review_20260925/` and record hashes of the relevant evidence. The quoted suggestion to delete these logs should not remove provenance needed for corrections.

## 3. First work package: correct the scientific record

**Target:** a defensible corrected manuscript and evidence package before adding new claims. Estimated focused effort: two to three working days, with chronology verification potentially extending this.

### Attribution and scope

Add a concise comparison of operators and implementation choices. With activity matrix H and pseudo-errors D, the core activity update is proportional to `D H^T (H H^T + lambda I)^(-1)`, after mapping normalization conventions. This is the FOOF activity operator with approximate rather than backpropagated errors. State differences in moment averaging, inverse refresh, damping normalization, norm matching, biases, and output-layer treatment. Do not claim those engineering differences establish novelty. [Benzing, ICML 2022](https://proceedings.mlr.press/v162/benzing22a/benzing22a.pdf).

K-nDFA has a KFAC-like two-sided structure, but its DFA pseudo-error moment is not automatically a Fisher factor. Avoid both “new natural gradient” and an unqualified claim of exact equivalence to standard KFAC.

Position the aligned linear model as explanatory background and a source of testable predictions. Amari et al. study how preconditioning interacts with signal/feature geometry and early stopping; Heckel and Yilmaz relate unequal learning times to early-stopping risk. The existing propositions should not be advertised as establishing those broad ideas for the first time. [Amari et al.](https://arxiv.org/html/2006.10732v3), [Heckel and Yilmaz](https://arxiv.org/pdf/2007.10099).

Credit alignment dynamics also has substantial prior work; the research target is the effect of conditioning on that process, not discovery of an alignment phase. Include [Refinetti et al.](https://proceedings.mlr.press/v139/refinetti21a.html), [Chu and Bacho](https://arxiv.org/abs/2306.02325), and initialization-dependent results of [Girotti et al.](https://arxiv.org/abs/2110.10815). Do not equate gradient alignment with sufficient performance or require forward-weight symmetry as a general condition.

Use [SKFAC](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_SKFAC_Training_Neural_Networks_With_Faster_Kronecker-Factored_Approximate_Curvature_CVPR_2021_paper.html) for the relevant low-rank Kronecker-factor inversion precedent. Check the claimed Zhang et al. ICML 2025 BN reference against its actual title and primary source before asserting a conflict; its identity was not established by this search. Compare normalization findings by architecture, FA versus DFA, learning rate, and protocol, rather than assume that different empirical outcomes contradict each other. [Liao et al.](https://ojs.aaai.org/index.php/AAAI/article/view/10279) already make normalization important in asymmetric feedback learning.

A closely related source already cited in the manuscript is [Boeshertz, Pascanu and Clopath, 2026](https://arxiv.org/abs/2606.11123), on rank collapse in FA and interventions using Muon and normalization. Turn that citation into a concrete distinction and, where applicable, a matched comparator before claiming novelty for improved update rank. Its FA results are not automatically DFA results.

Do not append all 27 or 34 references indiscriminately. Each addition must support a concrete sentence, method comparison, or baseline decision. Alphabetize and standardize the bibliography, verify the Dalm publisher metadata and Braun venue metadata, and preserve valid existing citations.

### Protocol and numerical corrections

1. Reconstruct the July 14 MNIST sequence from commands, raw records, source revisions, and time zones. Distinguish final-only evaluation in the later runner from earlier exposure of the same test set/seeds. Remove “fresh” and “test-blind” implications where unsupported. Verify the provenance of seeds 55–59 before calling them an unexposed extension. Keep Fashion and ReLU preregistration claims only within the scope established by their dated records.
2. Present the shared-rate synthetic suite as a geometry/stability intervention. Lead mechanistic comparisons with norm-matched results, while acknowledging that norm matching alone does not substitute for per-method tuning. Do not present large gains against high-loss DFA as clean evidence of improved generalization.
3. Label the tuned, test-selected BP study separately from the synthetic suite's own BP. State selection in the main figure caption, not only an appendix. Show task-aligned and long-horizon boundary results beside the positive regime interpretation.
4. Verify the reported ledger of 199 finite endpoints, 33 numerical failures, two structural failures, and six unlaunched trials. Separate missing infrastructure execution from algorithm failure. Never substitute successful seeds for failures.
5. Fix shared-seed wording for the longer budgets and identify hardware/precision for every timing cohort. Audit the ImageNet starred ZCA rate and overlap with evaluation seeds; label test-influenced choices explicitly. These block-output power-one-half results must remain distinct from activity-side inverse conditioning.
6. Replace ambiguous “Best NC” with explicit method names or a documented validation-only choice. Since the old vision archive has only full-rank feedback, remove the false rank-selection explanation. Use the separate validation-selected sweep only with actual training/validation counts and its own protocol.
7. Generate numerical tables from the named cohort. Verify the reported rounding corrections (+0.72 for the mixed BP-whitening difference; 0.71 and 0.10 SEM entries) rather than copying them without a reconstruction.
8. Add the four-rule Adam table as a test-selected exploratory control, naming the available diagonal inverse-square-root arm accurately. It does not isolate off-diagonal inverse-moment structure.
9. Expand the existing BN table with the within-BN comparison across all four regimes. Average feedback draws and designed cells within global seed before uncertainty calculations. Keep 28/32 cell wins descriptive. Omit legacy K+BN until scaling is verified.
10. Add one compact table for original/additional/pooled error-side cohorts, retaining the original Figure 3 cohorts. Do not present post-hoc pooled p-values as prospective confirmation.
11. Standardize uncertainty: identify independent units; show five seed values where practical; use explicit t intervals with their assumptions; remove uncorrected significance symbols; label crossed combinations “training runs.” Define all whiskers and bands consistently.
12. Name the task “ColoredMNIST-style” and state the single environment, clean labels, and train/test color correlations. Define gradient-like G versus descending g without changing correctly signed equations.
13. Correct the HHMI/Harvard affiliation grouping in the public build. Rebuild both conditional PDFs and both packages after the complete correction pass; do not upload the old September 19 package.

**Acceptance check:** every affected sentence links to a reproducible cohort and selection rule; no source claims a test-blind design contradicted by its history; original and pooled exports coexist; revised tables and captions agree.

### Preserve the earlier mathematical and presentation corrections

The preceding review cycles already repaired or targeted several issues. Treat these as regression checks during the rewrite, not reasons to restart the scientific structure:

- Keep exact conditional risk separate from the population-spectrum surrogate, and compare complete minimum risks rather than identifying avoided nuisance variance with total improvement.
- State the joint limiting condition using both rate ratios, or the sufficient condition `lambda_A/lambda_T -> 0` with `lambda_N/lambda_T -> infinity`. Letting the absolute damping vanish alone is insufficient when the spectrum changes.
- Separate scalar-output dynamics from first-layer gradient alignment. State that minimum trajectory risk and within-rule mode-timing ratios are invariant to global time rescaling; absolute flow times are not compute comparisons. B remains fixed: forward weights align to B.
- Preserve the norm-matching zero case and the correct order of the two noncommuting minibatch-space solves.
- Keep full-versus-diagonal claims about uncentered moments, disclose unequal historical searches, and avoid claims about all coordinatewise methods. E2 is the targeted experiment that can support a stronger interpretation.
- Keep Figure 1's reproducible initialization, step-size, damping, target, and moment protocol. Do not cite a fixed-condition-number panel as if it showed a condition-number sweep.
- Ensure the nuisance-energy correlation uses one identified result/cohort, or explains any difference between the earlier 0.677 and 0.62 summaries. Define the task-span basis U to have orthonormal columns.
- Make the study ledger disambiguate the earlier 280-model cohort and later 170- and 50-model follow-ups; “all models” must have a scope. Include hardware and precision rather than infer them from a partition name.
- Do not infer that spatial routing cannot help from two failed routing alternatives. Mark the proposed active-filter mechanism as a hypothesis. Describe the opposite CIFAR-10/CIFAR-100 clean-data effects without asserting an untested headroom explanation.
- State that the fixed-recipe loss instability affects raw DFA as well as conditioned DFA, and keep that intervention separate from independently accuracy-selected recipes.
- Keep uncertainty definitions beside the relevant graphics; show intervals if the discussion claims they are displayed. Label datasets and useful log-scale ticks. Remove duplicate panels only when they add no distinct evidence.
- Keep the supplementary roadmap and cross-references consistent with the final organization. Put essential cross-study reference tables near the associated text. Do not sacrifice readable type to force a float onto a page.
- An ethics section is optional and currently absent; preserve the separate required AI-use statement. Keep the abstract qualitative and avoid reinstating hardware-specific seconds.

## 4. Decisive new experiments, in priority order

### E1. Stable, tuned baselines and learning curves — highest priority

**Question:** is there a useful activity benefit after each method gets a competent optimizer, a suitable rate, and enough training?

Start with four predeclared synthetic cells spanning the existing regimes, the existing MNIST/Fashion factor settings, and the existing 3.68M-parameter CIFAR-10 MLP. Reuse data preparation and architectures to isolate the reason for differences. Do not rerun the entire 128-cell grid initially.

Core comparisons: tuned DFA, activity DFA, DFA+BN, activity DFA+BN, tuned BP, and FOOF-BP. Include FD-DFA in the practical comparison. Give BP and DFA access to suitable standard SGD/momentum or AdamW configurations under a declared development budget. Evaluate a Muon-based DFA comparator if the prior-work reproduction and implementation mapping are sound. Distinguish FA from DFA throughout.

There are two different comparisons: (i) matched implementation components and norm controls to isolate the operator; (ii) the best validated practical recipe for each algorithm. A published FOOF implementation and a component-matched BP+activity control may both be needed; forcing the published baseline into the nDFA recipe could weaken it.

Use validation to select rates, schedules, damping and checkpoints. A recommended initial development allocation is twelve configurations per method family on three development seeds, with the same staged expansion policy for boundary optima. Track actual tuning compute as well as trial counts. These are proposed settings to freeze in a protocol, not retrospective claims of preregistration. Select final replication size by precision for a prespecified practical effect, with ten independent model seeds as a useful small-model planning default. Average nested feedback draws first; do not extend sampling merely until a p-value crosses a threshold.

Measure accuracy **and** predictive loss, instability frequency, actual update counts, wall time, and memory. Separate numerical failures from finite high loss and from infrastructure failures. Any stability filters or loss constraints used for selection must be declared on development data and applied symmetrically, while retaining the original accuracy-selected analysis for transparency.

Train with validation-selected schedules until a prespecified plateau criterion or generous cap. For example, use a fixed tolerance over two evaluation windows and a validation-based continuation check; unresolved capped runs are not “converged.” Report complete curves at equal updates and equal measured work, final fixed-horizon endpoints, and validation-selected checkpoints. Do not stop every method at the fastest method's convergence point.

**Decision:** advance the mechanism and scaling claim only if a meaningful advantage remains, or if there is a reliable compute advantage that can be explained. If the benefit only rescues the old unstable rate, narrow the contribution immediately.

### E2. Separate mean activity, centered covariance, and scalar adaptation

**Question:** which part of the uncentered moment supplies the benefit?

For `M = Cov(h) + mu mu^T`, compare damped inverse operators based on:

- M (full current rule);
- diag(M) (matched inverse power, not inverse square root);
- Cov(h) (centered covariance);
- diag(Cov(h)) + mu mu^T (preserves the mean term while removing off-diagonal centered covariance);
- a trace-matched isotropic baseline plus mu mu^T (mean-direction control).

Use exactly the same forward network and update numerator for the primary comparison. Changing the forward activation by centering is a different intervention and should be an explicitly labeled additional control. Keep damping conventions, norm matching and tuning effort comparable; account for bias updates. Evaluate both with and without BN where feasible, because pre-ReLU BN does not zero the post-ReLU mean.

Measure the mean-energy fraction, centered spectrum, effective rank, and the mean direction's overlap with task-relevant directions in the synthetic generator. Freeze the corresponding probes for real data before confirmation. Include positive and task-aligned controls.

**Decision:** if the mean-preserving or rank-one control reproduces the gain, explain that result and evaluate the cheaper mechanism. If full centered correlations add a reliable benefit, support that narrower claim directly. No outcome is predetermined.

### E3. Explain the alignment phase — strongest potential distinctive contribution

**Question:** does conditioning improve how forward weights acquire useful credit, beyond a change in the instantaneous metric or step size?

At the **same checkpoints**, measure raw DFA-to-BP alignment and conditioned-update-to-BP alignment separately, per layer. Otherwise an immediate rotation by the preconditioner can be mistaken for improved learned alignment. Also record normalized directional derivatives, observed small-step loss changes, update norms, predictive loss and accuracy. BP diagnostics are offline measurements, not signals available to the local learner.

Predefine alignment onset (including handling runs that never align), the integrated negative projected update over a fixed early window, and a small number of primary outcomes. Compare matched update norms and stable tuned learning rates. Evaluate multiple feedback orientations/initializations and a held-out set of geometries.

Causal interventions: early-only versus late-only conditioning; freeze or swap the moment operator at a shared checkpoint; replay different operators against the same raw update; and change nuisance/task orientation while preserving the spectrum. Restore every state component before branch experiments. These tests distinguish instantaneous preconditioning from persistent representation changes. A correlation between alignment and accuracy alone is insufficient.

For variance, separate model initialization, data/minibatch order, and feedback randomness. The old compound feedback/order seed cannot identify pure feedback-matrix variance. Use a bounded crossed design on selected conditions and report the appropriate conditional or crossed-design uncertainty.

Develop theory for a coupled two-layer model **without imposing forward alignment at the start**. The desired result predicts an alignment transient or a failure boundary as a function of geometry and initialization. A one-step inner-product identity is a diagnostic starting point, not automatically a new theorem. Check the proposal against existing DFA dynamics and initialization theory, then test predictions on held-out simulations. If no useful new result emerges, retain the current aligned theory as background rather than overclaim its scope.

### E4. Resolve the error-side factor's role, with a fixed budget

**Question:** when is E genuinely non-scalar and useful, and when does retuning effectively turn it off?

Measure centered and uncentered error spectra, mean-error energy, effective rank, relative damping, and the deviation of the damped inverse from a scalar operator. Contrast one-vs-rest sigmoid and softmax losses under appropriate tuning. Extend damping grids past edge optima, including the exact A-only limit. Keep activity damping fixed for K-minus-A mechanism tests; a separate best-recipe comparison may retune both but answers another question.

Use a small controlled error-anisotropy intervention, matching total error energy and manipulating orientation independently of spectrum. The learner receives neither the planted noise covariance nor a clean target error. Compare against matched scalar scaling and simple common-mode controls. A known-feedback projection is an essential baseline when planted corruption falls outside the known feedback span; otherwise a trivial projection can masquerade as learned denoising.

Audit and reuse the existing teaching-channel prototype and its scope record before writing another simulator. Its implementation audit was not evidence that substantive training had already validated the hypothesis.

**Decision:** retain E/K in the main contribution only if the new evidence predicts and reproduces a useful non-scalar regime. Otherwise summarize the small-model result and damping boundary in the supplement, with a short main-text qualification. Do not launch another broad error-factor sweep without this mechanism.

### E5. One demanding application, after E1–E3

Choose one application where a competent DFA baseline is already established. Launay et al. cover neural rendering, recommendation, geometric learning and NLP; merely adding a Transformer is not itself a novelty claim. [Launay et al., NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/69d1fc78dbda242c43ad6590368912d4-Abstract.html).

Default first candidate: a reproduced neural-rendering task with deeper MLP credit assignment, which is close to the current method and gives a different application. A small Transformer is a conditional alternative if the implementation audit shows manageable feedback/moment costs and a credible baseline. Choose by a predeclared feasibility/protocol criterion, not by screening test outcomes for a positive result. Record unsuccessful pilots.

Compare tuned BP/AdamW, FOOF-BP where applicable, tuned DFA, activity DFA, and the strongest applicable decorrelation/optimizer control. Pin dataset splits, architecture, training scope, output loss, and local/BP module boundaries. A BP-pretrained frozen encoder with only a conditioned head is not evidence of training a deep network with local credit assignment.

Use two declared model sizes to test whether cost and benefit transfer. Include preprocessing, forward computation, feedback, moment updates/solves, optimizer work, memory, and precision in resource reporting. Repeatedly computing and discarding BP in the learner would invalidate a local-learning cost claim. Structured feedback, low-rank inverses, amortized updates, or mixed precision are explicit algorithm variants, not invisible implementation substitutions.

**Decision:** one convincing application with the mechanism and competitive baselines is more valuable than several weak demonstrations. If it fails, report the boundary; do not keep adding unrelated architectures.

## 5. A possible new idea, conditional on the mechanism

The near-term novelty target should be **an explanation with predictive value**. A new rule should follow only if the experiments identify a specific limitation.

Two bounded options are worth considering:

1. If mean-direction suppression explains the benefit, derive and evaluate a cheap diagonal-plus-low-rank conditioner. The matrix-inverse algebra is established; the research contribution would require a DFA-specific prediction and a demonstrated accuracy/cost advantage over BN, diagonal conditioning and full A.
2. If error anisotropy predicts a useful regime, evaluate a local statistics-based gate that enables E only when its anisotropy is reliable across independent training minibatches. Anisotropy alone cannot identify noise. Test task-aligned anisotropy as a negative control and compare with simply tuning the damping or disabling E.

Do not promise universal noise cancellation without assumptions. A learner cannot generally distinguish signal from nuisance solely from activity variance. The earlier teaching-residual proposal already has a [prior-art and algebra audit](ndfa_neuron_teaching_prior_art_2026-09-14.md): some apparently new versions collapse exactly to ordinary activity conditioning plus scalar/row rescaling. Any continuation must survive those equivalence controls and teacher-shuffling tests before receiving training resources.

Keep the dendritic interpretation accurate: layer-local availability of activity and feedback does not by itself make dense covariance inversion synapse-local or establish a cortical implementation. A circuit approximation and its information/communication costs would be a separate biological contribution. Do not add an unsupported neuroscience application simply to change the submission category.

## 6. Paper structure, figures, and supplement

Revise the story after the E1–E3 decision, not after every pilot. A provisional title is **“Activity Geometry and Credit Assignment in Direct Feedback Alignment.”** Use it only if the credit-assignment mechanism is supported; otherwise retain the current neutral title or choose a title centered on stability and efficiency. Do not keep “error geometry” prominent solely to imply a robust second contribution.

The abstract should state the credit-assignment problem, acknowledge established conditioning, identify the specific new finding, give the validated scope, and acknowledge the strongest remaining limitation. Use qualitative comparisons and at most one informative quantitative anchor. Avoid hardware-specific seconds and a list of small benchmark gains.

Suggested main-text order: problem and prior-art boundary; precise rule and computational/locality assumptions; stable-baseline result; mechanism and geometry controls; one practical application; scope and limitations. Introduce every symbol and diagnostic before use. Start each section with the question it resolves. Move development history into the protocol ledger, while retaining selection disclosures where needed to interpret a result.

| Figure role | Useful panel content, conditional on results |
|---|---|
| 1: explain the method and hypotheses | Credit pathway, activity/error operators, mean versus covariance distinction, one clearly scoped theoretical illustration. |
| 2: establish the reliable effect | Tuned learning curves, equal-work comparison, long-horizon endpoints, normalization/norm control. |
| 3: test the alignment mechanism | Raw alignment over training, paired onset/negative-projection summary, a causal early/late or swap intervention, separated seed variability. |
| 4: identify the relevant statistic | Full/diagonal/mean-preserving controls and their predictions; add error-factor panels only if they answer the same mechanistic question clearly. |
| 5: demonstrate practical relevance | Application performance versus work, scale transfer, memory/overhead, and a useful robustness or replication panel. |

Preserve the user's compact layout preference: four panels in one row when legible, short Figure 3, and useful extra Figure 5 panels. Do not force four panels if that produces tiny labels or padding. Keep panel letters above and left of all panel content, aligned across rows; use shared axes where appropriate, minimal in-panel prose, external compact legends, vector output and readable final-size labels. View both standalone assets and the rendered manuscript pages. Define every interval in the caption. A Times-font conversion is optional; readable consistent plots take priority.

Curate the supplement by scientific role:

1. Definitions, exact derivations, assumptions and proof details.
2. A short study/cohort/selection/hardware ledger and reproducibility instructions.
3. Stable-baseline and activity-mechanism controls.
4. Error-factor results and original/extension provenance.
5. Practical/application protocols and complete primary endpoints.
6. Compact boundaries and robustness needed to interpret main claims.

Use approximately 15–25 supplementary pages as an editorial target, not a hard quota. Keep counterevidence that limits a main claim; remove duplicate plots, superseded development tables, command logs, and unrelated pilots from the reader-facing PDF. Preserve all relevant records in a versioned archive with a clear index. Every retained subsection should support a main claim, explain a necessary protocol, or document a material boundary. Keep floats within their study using selective barriers, without splitting sentences or leaving avoidable half-empty pages.

## 7. Common experimental and release requirements

- Freeze development and confirmation seeds, split hashes, source revisions, candidate grids, selection rules and intended contrasts before each new confirmation. Historical benchmark exposure must remain disclosed; new seeds do not erase it.
- Register a practical-effect or precision target, not only a significance target. Report all planned arms. Use paired differences and confidence intervals, correct declared testing families, and distinguish designed-condition variation from replication uncertainty.
- Use `kempner_h100_priority`, with `kempner_eng` and `kempner_requeue` also authorized on September 25 to accelerate independent rounds. Keep hardware fixed within each timing comparison and checkpoint requeue jobs. Estimate actual GPU hours from a bounded timing pilot. Keep each work package's case count and allocation explicit; current assignments are documented in [the parallel development protocol](ndfa_parallel_development_20260925.md).
- Do not invent an overall GPU-hour promise before timing. A reasonable planning allocation is roughly half the experimental effort on E1, a third on E2–E3, and the remainder on E4 plus feasibility work; expand E5 only after the earlier decision. Calendar estimates depend on queue access and implementation complexity.
- Export separate original, added and pooled summaries. Table generators must select named cohorts rather than all files matching a wildcard. Preserve failures, seeds and prior results immutably.
- Bind each numeric table, plotted point and caption to source inputs and aggregation definitions. Use targeted tests for operator equivalence, cohort separation, error scaling and selection leakage; retain the existing successful mathematical/dual-solve checks.
- Review all identifying content in the anonymous PDF and ZIP, including metadata, local paths, logs, acknowledgments and links, while retaining required third-party attribution. A public arXiv version is not itself an anonymity violation.
- Rebuild both PDFs, arXiv source and the anonymous ZIP from clean extraction. Check unresolved references, overfull boxes, page limits, figure readability and package-size limits. Preserve the AI-use disclosure and correct public affiliations. Do not treat a passing package verifier as proof that every scientific interpretation is correct.

## 8. Schedule and checkpoints

| Window | Work and concrete deliverable | Decision before spending more |
|---|---|---|
| Days 1–3 | Chronology/citation corrections, cohort exports, corrected tables, fixed baseline recipes, study ledger | Corrected claims agree with evidence; new comparisons and tuning budgets are frozen. |
| Remainder of week 1 | E1 development and stable-baseline learning curves; instrument shared-checkpoint probes | Is there a meaningful residual benefit or credible efficiency result? |
| Week 2 | E1 confirmation and E2 mean/covariance controls; E3 interventions and initial theory | Which mechanism survives, and can it predict held-out behavior? |
| Week 3 | Complete E3; bounded E4; reproduce and time one application | Select the final contribution; stop unsupported branches. |
| Week 4, if justified | E5 confirmation, coherent rewrite, five-figure design, curated supplement and rebuilt releases | Every main claim has direct evidence and closest-prior-art comparison. |

This is a provisional three-to-four-week research schedule, not a promise that a new theorem or positive scaling result will appear. If E1 invalidates the broad claim, narrow the paper promptly instead of running the full schedule.

## 9. Submission decision

An ambitious conference submission becomes more defensible if the revision delivers: (i) an effect against competent baselines, (ii) a distinctive and tested explanation or method improvement beyond established preconditioning, and (iii) one credible transfer/application result, with complete provenance. These are project decision criteria, not an invented conference acceptance checklist.

A carefully scoped empirical/mechanistic paper remains valuable if only some of those aims succeed. TMLR evaluates supported claims and reader interest and does not require method novelty; it does not accept work merely because code runs correctly. [TMLR acceptance criteria](https://www.jmlr.org/tmlr/acceptance-criteria.html). Choose the venue after the early evidence decision rather than attach an uncalibrated acceptance percentage now.

The current ICLR 2027 official full-paper deadline is September 25, 2026 AoE, after a September 18 abstract deadline. A several-week revision cannot be presented as an improvement completed before that deadline. If an active submission exists, any decision about its status belongs to the authors; this planning task does not submit or withdraw it. The guidelines also distinguish withdrawal before the paper deadline from withdrawal afterward, so the review's blanket statement about immediate public withdrawal needs that qualification. [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

**Immediate implementation order:** provenance and attribution corrections → explicit cohort/selection tables → stable tuned baselines → mean/covariance and alignment mechanism → conditional error-factor/application work → final narrative and release rebuild.

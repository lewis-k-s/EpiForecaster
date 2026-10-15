# Loss-type autoresearch plan

Date: 2026-09-18. Status: proposed protocol; execution has not started.

Working branch: `codex/loss-type-research`.
Inspected source: `df2dfd7cbc34ebaa51cacaf829c33279a11fd6d3`.
This plan does not create a goal, submit jobs, or set a compute budget.

## 1. Research question and scope

Which training objective and predictive distribution produce useful, calibrated
forecasts for cases, deaths, hospitalizations, and wastewater? Does the result
also improve the probability and classification of high-incidence events?

Separate three effects: the training score, the distribution family, and
calibration after training. Compare each trained candidate with both the raw
Normal-NLL baseline and a calibrated version of that same baseline. A useful
result can be a better loss, a calibration-only solution, or evidence that no
candidate gives a reliable gain.

Keep the model backbone, SIRHD dynamics, observation inputs, delay kernels,
embedding checkpoint, data, and supervision masks fixed. Dynamic mobility and
mobility off are two declared comparison conditions. Do not search mobility
weights or redesign the graph in this study. Do not begin with architecture
search, broad HPO, learned degrees of freedom, or per-head loss combinations.

The pasted R1/R2 account motivates this work: it reports undercoverage and heavy
residual tails, especially for hospitalizations and wastewater, and benefit from
validation-fitted scale correction. These are supplied historical findings,
not measurements reproduced on this branch. The cited September 17 report is
not tracked at the inspected commit. Recover its immutable run identities
before using its numbers in the final evidence record.

## 2. What exists and what must change

The following facts were checked in the current source.

| Component | Current behavior | Required work |
| --- | --- | --- |
| `models/configs.py`, `evaluation/losses.py` | Normal NLL, Normal CRPS, and fixed-df Student-t NLL exist. Student-t CRPS is rejected. One observation objective applies to all heads. | Add and validate only the objective options used in each stage. |
| `evaluation/metrics.py` | NLL and 50/80/95 intervals support both families. CRPS is Normal-only; Student-t CRPS is missing. | Add Student-t CRPS, a common WIS implementation, CDF diagnostics, and invalid-prediction counts. |
| `training/epiforecaster_trainer.py` | Normal probabilistic runs prefer mean validation CRPS. Student-t NLL falls back to mean NLL. Missing heads can be omitted from the mean. | Use an explicit, common selection metric and required-head contract. Missing required scores must stop selection. |
| `evaluation/aggregate_export.py` | The main per-head aggregate table contains point metrics and observation counts, not the probabilistic scorecard. | Extend canonical aggregates before campaign ranking. |
| `evaluation/granular_export.py` | Schema 3 has NLL, CRPS, and three intervals. It lacks the full distribution, event probability, and event label contract. | Version the export and retain all fields needed to audit scores and events. |
| `scripts/hpo/ablation_worker.py`, `scripts/analysis/analyze_ablations.py` | Campaigns use an exact condition/fold/seed matrix and currently consume test aggregate artifacts. | Add a declared development evaluation role. Prevent adaptive decisions from reading final-holdout artifacts. |
| `data/dataset_factory.py` | Persisted CV rotates adjacent node folds through validation and test. | Audit shared observation sources and temporal overlap. Spatial CV alone does not establish future-time performance. |

Use the fixed settings in
`configs/campaigns/epiforecaster_nood_region2vec_context_v1.yaml` as the proposed
model anchor: OD-free Region2Vec, all four MLP delay kernels, and the existing
50-epoch ceiling. Verify its DVC inputs before adopting it. Do not silently
replace a missing embedding with a different one. This is a new matched study;
it is not a claim to reproduce the later R1/R2 deployment.

## 3. Freeze the evaluation protocol first

### Data roles and leakage control

Create a versioned protocol manifest with four explicit data roles:

1. **Fit:** model training and all fitted preprocessing. Estimate event-value
   thresholds from this role only.
2. **Selection:** checkpoint selection, candidate ranking, and loop decisions.
3. **Calibration:** fit scale corrections and decision thresholds after a
   checkpoint is frozen. Score their benefit on data outside this fit role.
4. **Final holdout:** evaluate the frozen finalists once, after search stops.

Use blocked or source-grouped inner splits to separate selection and
calibration. If data are limited, use a fixed cross-fitting design within the
development data. Never fit a calibrator and report its in-sample coverage as
validation evidence. Record the exact membership of every role.

Treat all folds and periods used in the previous research as development data.
The rotating ten-fold scheme means a region can be validation data in one run
and test data in another. A `test` filename does not make it an untouched final
holdout. Freeze a common outer time block or source group that is excluded
from **all** candidate fitting, selection, calibration, and threshold fitting.
Audit prior use before describing this block as fresh. If no unused block is
available, report retrospective confirmation and state the remaining need for
prospective evidence.

For temporal evaluation, require every training label to precede the cutoff.
Exclude overlapping label windows at role boundaries, with a gap of at least
the 28-day forecast horizon, extended for known reporting delay. Historical
inputs must be available at the forecast origin. Current revised data are not
automatically an as-of data stream; record any lack of vintage information.

Check whether multiple regions share the same observed source series. Use
source groups for leakage checks and uncertainty estimates. Do not count
duplicated source measurements as independent evidence. Preserve the current
observed-only masks and reporting cadence. Missing or imputed targets do not
become negative outbreak labels.

### One score for selection across all candidates

Use **equal-head mean weighted interval score (WIS)** in the existing
`log1p(per-100k)` space as the proposed loop and checkpoint score. Lower is
better. Use the same score for NLL, CRPS, WIS, and later quantile candidates.
Require all four declared heads on the fixed eligible evaluation support.

WIS permits comparison of full distributions and quantile forecasts. It also
accounts for interval width and missed observations. Use a fixed set of 23
quantiles: `0.01, 0.025, 0.05, 0.10, ..., 0.95, 0.975, 0.99`, where the middle
sequence has step 0.05. These give 11 central intervals plus the median. Use
the standard interval weights `alpha/2` and median weight `1/2`.
[Bracher et al. describe this score for epidemic forecasts.](https://arxiv.org/abs/2005.12881)

This changes the current checkpoint rule. Apply it to a freshly trained
Normal-NLL control as well as every candidate. Also retain the best-CRPS
checkpoint from each parametric run as a declared selection-policy sensitivity
analysis, without using that analysis to select the winning loss. Do not
attribute a comparison with historical CRPS-selected runs to training loss
alone. Freeze this choice before the first scientific run.

For each head, average over the same eligible forecast rows. Then average
heads equally, followed by equal weighting of the declared fold/seed pairs.
Report the individual heads and horizon bands; a pooled mean cannot hide a
regression in a weak head. Add source-balanced and region-balanced sensitivity
summaries without changing the primary estimator during the loop.

### Canonical scorecard

| Purpose | Measures |
| --- | --- |
| Distribution quality | WIS and its width/underprediction/overprediction terms; exact family CRPS where supported; NLL for density models only |
| Calibration | 50/80/95 coverage, lower and upper miss rates, PIT or quantile-reliability summaries, and calibration sample counts |
| Sharpness | Mean and median interval widths at each level, shown beside coverage |
| Point accuracy | MAE of the predictive median; RMSE as a secondary descriptive metric |
| Event quality | Brier score, reliability bins, average precision, precision/recall, F1/F2, TP/FP/TN/FN, event prevalence, and false-alert rate |
| Stability and cost | Non-finite predictions, scale-floor hits, scale/tail extremes, gradient norms, selected epoch, updates, elapsed time, GPU-hours, and failure rate |

Report by head, horizon days 1-7/8-14/15-21/22-28, fold, seed, source group,
observation-support stratum, and mobility condition. Use narrower slices only
with adequate support. Return null plus counts/reason for unsupported metrics.

Keep score units explicit. Log-space CRPS is not incidence-space CRPS. Apply
the monotone inverse transform to quantiles for original-unit interval and
median summaries. Do not treat `expm1(location)` as a predictive mean. For an
unbounded Student-t variable in log space, its exponential has no finite first
moment; finite log-space variance does not fix this. Thus original-unit mean
or ordinary CRPS is not a valid Student-t endpoint here. Finite-quantile WIS
remains available. This restriction follows from the polynomial Student-t
tail and exponential inverse transform. Also audit probability below zero in
log space. Do not silently clip distributions or intervals during scoring.

## 4. Outbreak classification contract

Until a public-health definition is supplied, use **high-incidence exceedance**
as a research proxy. It is not a verified outbreak diagnosis or an operational
alert rule. Make cases the primary event endpoint and hospitalizations the
secondary endpoint. Wastewater exceedance is a separate signal metric.

For one eligible target observation, define:

```text
event = 1 if observed_target > event_threshold else 0
event_probability = 1 - forecast_CDF(event_threshold)
predicted_event = 1 if event_probability >= decision_threshold else 0
```

Proposed value threshold: the 90th percentile of unique observed values in
the fit data, per head and in its native target units. Count each underlying
source/date once when fitting it. Freeze a pooled training threshold for
unseen regions; do not use a held-out region's future history to create a local
threshold. Document any group-specific thresholds and minimum support before
use. Use strict `>` for ties. The empirical event frequency need not be 10%.
Use the 95th percentile only as a predeclared sensitivity analysis.

Score observed daily case targets and the actual hospitalization reporting
targets. Check whether a weekly hospital observation is a weekly sum, rate, or
snapshot before naming it. Do not create a weekly-total probability by adding
daily quantiles. A probability of "any outbreak in the next week" requires a
joint forecast or a separately defined aggregate target; marginal forecasts
alone do not identify it. Defer onset, persistence, and lead-time metrics until
that joint event contract exists.

Report precision and recall at `p >= 0.5` for a common fixed decision rule.
Also report a calibrated operating rule: choose the threshold on calibration
data to maximize recall subject to a predeclared minimum precision and minimum
alert support. Freeze it for evaluation. A provisional precision floor is
0.50; it is a research setting, not a public-health recommendation. Include
false alerts per 100 eligible opportunities and the achieved precision on the
evaluation set; the calibration constraint is not a guarantee out of sample.
If no threshold qualifies, record "no feasible operating point".

Use average precision (AP) as the specified precision-recall summary; do not
label a trapezoidal PR area as AP. Include the prevalence reference. AP alone
does not measure probability calibration, so retain Brier score and reliability
bins. [The scikit-learn definition specifies the AP calculation.](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

If there are no positives, recall and AP are unavailable under this research
contract. If there are no predicted positives, precision is unavailable. Retain
the confusion counts. Use the same finite eligible rows for each paired model;
invalid predictions must be visible and must not improve the score by reducing
the denominator. Forecasts at different leads are distinct opportunities, but
they are correlated. Use fixed origin schedules and block uncertainty estimates.

## 5. Ordered experiments

### Stage A: calibration controls without new training

For a provenance-matched baseline checkpoint, export selection and calibration
forecasts through the canonical evaluator. Apply these frozen postprocessors:

- C0: identity; no calibration.
- C1: one positive scale multiplier per head, fitted for central 80% coverage.
- C2: one positive scale multiplier per head, fitted to minimize WIS.

Fit separately for each model/fold/seed, use declared bounds, and store support
counts. Verify on selection data or fixed cross-fitted predictions. Report all
three interval levels: matching 80% coverage does not establish full calibration.
If historical checkpoints cannot meet the new data-role contract, use them only
for diagnostics and repeat the controls on Stage B's fresh baseline.

Later, permit a regularized head-by-horizon correction if a global correction
fails consistently. Conformalized intervals are a separate candidate, with
explicit dependence and shift assumptions. Do not claim ordinary exchangeable
conformal guarantees for overlapping epidemic windows.
[Gibbs and Candes discuss adaptive coverage under distribution shift.](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)

### Stage B: matched objective-by-family screen

| ID | Distribution | Training objective | Question |
| --- | --- | --- | --- |
| B0 | Normal | NLL | Fresh baseline under the common protocol |
| B1 | Normal | CRPS | Does the gradient from CRPS improve useful uncertainty? |
| B2 | Student-t, df=5 | NLL | Does a heavier-tailed family explain the error shape? |
| B3 | Student-t, df=5 | CRPS | Is there an interaction between score and family? |

Cross these four settings with dynamic mobility and mobility off. Retain the
same OD-free region embedding in both. Proposed initial folds are `[1, 4, 5]`,
with paired seeds `[52, 53]`: **4 x 2 x 3 x 2 = 48 training identities**.
These are spatial development screens, not final confirmation.

B3 requires a tested differentiable Student-t CRPS. A closed-form reference
exists, but the current code does not implement it. Validate values and
gradients, including tail limits, before scheduling it. If a numerical method
is used, bound its approximation error and runtime. Do not substitute a Normal
score or an unvalidated sample estimate under the same condition ID.
[Jordan et al. provide the scoringRules reference.](https://arxiv.org/abs/1709.04743)

Keep fixed head weighting, optimizer, learning rate, batch schedule, training
ceiling, and early-stop rule. Scores have different magnitudes, so record head
gradient norms and actual updates. This first result is the loss choice under
one fixed optimizer policy. A later, equal-budget learning-rate check can
separate optimization mismatch from a poor objective. Do not silently tune
only the leading candidate. Student-t scale is not its standard deviation;
at fixed df, report both using `sd = scale * sqrt(df / (df - 2))`.

Apply C0/C1/C2 to every completed candidate. Compare raw with raw and calibrated
with calibrated, then compare the complete candidate pipeline with calibrated
Normal NLL. This tests whether retraining adds value beyond postprocessing.

### Stage C: bounded, evidence-driven extensions

Run at most four new settings in the first adaptive budget. Choose them from
the following ordered menu based on development results. Record the choice and
hypothesis before each run. Start in the fixed dynamic-mobility condition; any
finalist must later be checked with mobility off.

| Evidence | Next candidate | Isolation rule |
| --- | --- | --- |
| 80/95 intervals remain poor | Parametric WIS training using the common quantiles | Retain the current best family and architecture. |
| Systematic asymmetry or shape error persists | Ordered quantile head trained with pinball/WIS | Declare a representation change; enforce noncrossing and retain the SIRHD location pathway. |
| High-incidence probabilities remain poor | Base proper score plus threshold-weighted CRPS, or base score plus Brier loss on CDF-derived event probabilities | Freeze event definition and positive mixture weights. Keep a nonzero whole-distribution score. |
| Student-t wins but df sensitivity is unclear | Fixed df in `{3, 10}` around the df=5 control | Count each df as a separate candidate; do not learn df in this round. |
| Head gradients show imbalance | CRPS or WIS with GradNorm | Treat weighting as a separate factor. NLL plus GradNorm remains disallowed. |
| One loss has clear optimization failure | Small matched learning-rate comparison | Give the baseline the same extra search budget. |

Threshold-weighted CRPS can emphasize the upper tail while retaining a proper
score. Arbitrary weighting by the realized outcome, focal classification loss,
or class-weighted BCE can change the desired probability target. They are not
first-round calibrated-probability objectives.
[Gneiting and Ranjan give the weighted-score framework.](https://stat.uw.edu/research/tech-reports/comparing-density-forecasts-using-threshold-and-quantile-weighted-scoring-rules)

For quantile-only forecasts, NLL is unavailable. A finite quantile grid also
does not uniquely determine event probabilities. Before admitting that
candidate, specify and test its monotone CDF interpolation and tail treatment,
or emit event probabilities through a coherent additional output contract.
WIS stays the common selection score; CRPS is not renamed from a coarse WIS.

### Stage D: frozen confirmation

Freeze the baseline and one finalist, including calibration and alert rules.
Expand to all ten development folds and seeds `[52, 53, 54]` in both mobility
conditions: **2 x 2 x 10 x 3 = 120 identities before valid reuse**. This checks
geographic and stochastic stability; it does not remove prior selection bias.

Then evaluate the two frozen pipelines on the reserved outer temporal/source
holdout. For one outer split, two mobility conditions, and three paired seeds,
allow up to **12 final training identities**, if refitting is required. Keep a
distinct inner calibration set. Record every data role again during refit.
Do not use the final holdout to choose a different loss, calibrator, or alert
threshold. A failed confirmation ends with a negative or inconclusive result.

The proposed envelope is 48 screening identities, at most 24 adaptive identities
(four settings x three folds x two seeds), up to 120 expanded identities, and up
to 12 outer-split identities: **204 before reuse**, plus bounded acceptance runs.
This is a planning count, not an approved allocation. Set the GPU-hour and wall
clock caps when creating the goal. Measure run cost before choosing the budget;
reserve confirmation capacity before spending the adaptive budget.

## 6. Promotion criteria and uncertainty

Use WIS as the one scalar loop ranking value. Apply calibration, event, and
per-head limits as separate gates. Do not optimize a changing weighted sum of
metrics, precision alone, or coverage alone. Proper scores and calibration
diagnostics answer related but different questions.
[Gneiting and Raftery establish the proper-score basis.](https://doi.org/10.1198/016214506000001437)

The following are proposed research tolerances to freeze at goal setup:

- At least 2% lower mean WIS than the relevant baseline pipeline.
- No head has more than 2% worse WIS or median MAE.
- Central 80% coverage is within 3 percentage points of nominal per head;
  central 95% coverage is within 2 points. Report horizon errors separately,
  with a proposed maximum 5-point error in sufficiently supported bands.
- Cases-event Brier score does not worsen by more than 2%. At the declared
  operating rule, recall does not fall by more than 2 percentage points while
  meeting the specified precision/support requirement.
- All required outputs are finite and complete, with no unresolved numerical,
  provenance, support, or data-role failures.

Use absolute differences when a relative-score denominator is zero or too
small under a predeclared tolerance. Sparse event strata can make a gate
inconclusive; they do not count as a pass. During screening retain promising
but uncertain candidates for the bounded confirmation stage. Do not weaken
the gates after seeing results.

Estimate paired differences on matched forecast support. Use source-group and
temporal-block resampling; group all predictions of the same observed outcome
together. Block lengths must cover forecast overlap and account for epidemic
serial dependence. Report seed variation separately. Do not treat millions of
forecast rows, overlapping CV training sets, or folds that share sources as
independent replicates. The analysis must produce tested canonical aggregate
and uncertainty artifacts; plots only read them.

A confirmed superiority claim requires both the practical gain and a paired
95% uncertainty interval that supports improvement on the frozen final
endpoint. Evidence for noninferiority must account for the stated margins,
not only compare point estimates. If fresh independent support is unavailable,
limit the claim to retrospective evidence. Track all attempted candidates to
make adaptive selection and multiple comparisons visible.

## 7. Implementation work packages and exit gates

| Package | Main files or outputs | Exit gate |
| --- | --- | --- |
| P0: freeze protocol | New versioned research manifest; campaign anchor; split/source and prior-run identity audit | All data roles, thresholds, metrics, budgets, and baseline hashes are explicit. |
| P1: canonical scoring | `evaluation/metrics.py`, `eval_loop.py`, `aggregate_export.py`, `granular_export.py`, `epiforecaster_eval.py` | WIS, CRPS by family, coverage, PIT, event metrics, counts, and exports agree on fixed fixtures. |
| P2: objectives and selection | `models/configs.py`, `evaluation/losses.py`, `training/epiforecaster_trainer.py`; observation heads only when needed | Valid values and gradients; one explicit checkpoint score; required heads enforced; save/load/resume retain identities. |
| P3: calibration and data roles | New tested calibration/event modules under `evaluation/`; `data/dataset_factory.py` where required | No fit/evaluation role overlap; thresholds and calibrators round-trip with hashes and membership. |
| P4: campaign control | New `configs/campaigns/loss_research_*.yaml`; supported MN5 template; ablation worker/analyzer and workflow contracts | Exact matrix, allowed config changes, development-only decisions, complete probabilistic aggregates, and restart receipts. |
| P5: experiments and closeout | Immutable MN5 runs; canonical scorecards; research ledger and final report | Every attempt classified; paired evidence validated; winner or negative/inconclusive result documented. |

Schema and defaults belong in dataclasses. New output filenames, meanings,
quantile levels, score units, calibration IDs, and event-definition IDs must be
versioned before launch. Export location, scale, family, df, quantiles, target
value, source/region, origin/target dates, lead, mask/support, data role, and
event label/probability/decision. Extend the evaluation manifest and workflow
output contracts to hash these artifacts. Prevent the finalizer from accepting
an old MAE-only table as a complete loss-research result.

Tests must cover numerical references and gradients; Normal limits of Student-t;
extreme scales and tails; exact WIS/pinball identities; CDF event probabilities;
missing masks; all-zero/all-one events; threshold ties; no alerts; quantile
ordering; data-role isolation; calibration persistence; checkpoint selection;
paired support; canonical aggregation; export schemas; and workflow completion.
Test enabled/disabled feature paths and material interactions. Invalid
predictions must be detected before sanitization can conceal them.

Run focused tests while implementing, then `uv run ruff check .` and
`uv run pytest tests/` before scientific execution. Run best-effort Pyright.
Use only the existing bounded local smoke interface for local training. Add a
small MN5 train/evaluate acceptance matrix before the 48-identity screen. Full
production settings must not run locally. No exhaustive plot tests are needed.

## 8. Autoresearch loop contract

Use the existing immutable MN5 Snakemake workflow. The loop chooses and records
experiments; it does not introduce a second training launcher or turn Optuna's
coordination journal into an undeclared random search.

```text
Read protocol, budget, ledger, and canonical completed evidence.
Choose one unresolved hypothesis from the ordered candidate menu.
Write candidate ID, parent, expected effect, exact patch/config, and budget.
Validate code, config differences, score contracts, and acceptance tests.
Prepare a new immutable run; dry-run; submit through mn5_workflow.sh.
Wait for scheduler completion and verify required scientific artifacts.
Calculate paired development evidence using the frozen evaluation protocol.
Classify: promote / reject / inconclusive / implementation failure / infra failure.
Append evidence, costs, and the next permitted decision to the ledger.
Stop or repeat within the fixed budget; final confirmation runs only once.
```

Each ledger entry includes Git and bundle hashes, config/campaign hashes, DVC
and environment identities, fold/seed/split roles, checkpoint-selection score,
calibration/event versions, attempts, artifact hashes, resource use, decision,
and reason. Keep rejected and failed candidates. An infrastructure retry keeps
the scientific identity; a code or config change creates a new identity and a
new immutable run. Existing run bundles and results must never be edited.

An incumbent changes only after a complete matched comparison. Partial output
can diagnose failures but cannot establish a winner. Reuse completed artifacts
only when source, environment, inputs, protocol, and output hashes match; the
same filename or Git branch is insufficient.

Stop when the predeclared candidate/GPU-hour/wall-time cap is reached, when
three consecutive complete adaptive comparisons give no qualifying improvement,
or when frozen confirmation is complete. Two repeats of the same causal
numerical or infrastructure failure stop that candidate path for diagnosis.
An unresolved data leak, score mismatch, or invalid provenance blocks scientific
promotion. Budget exhaustion is not evidence that the research objective was
achieved. Allow "no better loss than calibrated Normal NLL" as a valid result.

If a later goal authorizes unattended work, use one task heartbeat tied to the
run state. Notify on a meaningful result, failure, completion, or required user
action. No goal, heartbeat, or remote work is created by this planning task.

## 9. Goal setup and final deliverables

Before launching the loop, freeze the proposed protocol choices or replace them
with explicit user choices: compute and time caps, WIS selection, the event
definition, research precision floor, practical improvement limits, and the
outer holdout. Establish DVC availability and baseline provenance. These are
the entry gates for execution, not reasons to start jobs during planning.

The final deliverable is a reproducible recommendation with:

- the selected loss/family/calibration pipeline, or a clear negative result;
- raw and calibrated comparisons with matched Normal-NLL controls;
- per-head and per-horizon scores, calibration, and interval widths;
- event reliability, precision/recall, AP, Brier score, support, and false alerts;
- paired uncertainty, sensitivity results, all attempted hypotheses, and cost;
- tested code, declarative configs, immutable run/artifact identities, and a
  report that states the limits of retrospective and source-shared evidence.

Production-default changes and operational alert deployment are separate from
the research result. Keep the chosen research configuration explicit and
reviewable before any later cutover.

## 10. Post-loop requirements for successor campaigns

The completed `loss-type-research-v1` protocol and its artifacts remain frozen
historical records. The following requirements supersede its three-fold Stage B
design for any new loss, mobility, or other EpiForecaster ablation:

1. **Campaign-only execution.** Every scientific ablation is a checked-in
   declarative campaign under `configs/campaigns/` and runs through the canonical
   MN5 ablation workflow. Per-run template generation is limited to bounded
   acceptance or exact historical reproduction and cannot select a winner.
2. **Complete spatial coverage from the first scientific estimate.** Every
   condition and paired seed runs all ten folds in
   `representative_10fold_mobility_refined.json`. All 161 valid target regions
   therefore enter the held-out test role exactly once per condition and seed.
   A fold subset cannot be used to rank or eliminate an ablation.
3. **Complete declared test support.** Each fold is evaluated over every
   eligible target date, forecast horizon, observed head, and target region in
   the campaign's declared held-out test set. Regional or temporal subsampling
   is permitted only for scientifically ineligible smoke runs.
4. **Matched completeness before ranking.** Analysis starts only after the exact
   `condition × fold × paired-seed` identity matrix is complete and baseline and
   candidate outputs have matched observed support. Missing regions, dates,
   horizons, or heads make the comparison incomplete rather than silently
   changing its population.
5. **Heterogeneity is a required result.** Report the aggregate endpoint and
   paired deltas by target, region, horizon, fold, and seed. Include the
   distribution of regional effects and the fraction of eligible regions
   improved or worsened. Mobility comparisons additionally predeclare
   mobility-context strata so that gains concentrated in strongly connected
   regions are not hidden by a pooled mean.
6. **Calibration remains out of sample.** A calibrator or decision threshold is
   never fitted and scored on the same observations. Use a separate declared
   calibration role or a frozen cross-fitting design without shrinking the
   scientific evaluation to a selected spatial subset.
7. **Claims match the evaluation population.** Full-fold retrospective
   evaluation supports conclusions about the complete declared historical test
   set. A separate outer temporal holdout is required only for a confirmation
   claim and must be named separately; a small development slice must not be
   described as the full test set.
8. **Dependence-aware uncertainty.** Full regional coverage does not make
   municipalities or folds independent. Retain temporal blocking and shared
   source grouping for intervals. Region-level effects are descriptive unless
   an explicit dependence model supports inference.

The successor loop cannot launch until the campaign analyzer can consume the
probabilistic endpoint used for selection and emit the required region-level,
head-level, horizon-level, fold-level, and seed-level matched evidence. The v1
point ranking remains useful for choosing comparison arms, but it is not a
full-coverage ablation result.

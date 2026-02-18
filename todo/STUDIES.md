## Studies Plan

Goal: produce credible evidence that the joint inference-observation model improves forecasting and latent-state inference under sparse, post-testing data (2022-2026).

### Study 0: Evaluation Protocol Lock-In (must run first)

- [ ] Freeze dataset slices and masks for all studies:
  - [ ] `dense-supervision` era: period with WW + cases + hosp (+ deaths where available)
  - [ ] `sparse-supervision` era: period with WW-dominant signals and missing/low-quality case reporting
  - [ ] `post-cases` era: train up to full case era, continue into WW-dominant regime
- [ ] Define one canonical split strategy for time series:
  - [ ] rolling-origin backtest windows (e.g., 4-8 folds)
  - [ ] fixed forecast horizons: 1, 2, and 4 weeks
- [ ] Lock reporting metrics:
  - [ ] Forecast metrics per observed head: MAE, RMSE, MAPE/sMAPE, correlation
  - [ ] Probabilistic calibration if available: CRPS, PI coverage
  - [ ] Latent plausibility metrics: mass conservation error, smoothness/TV penalty, effective reproduction trend consistency
  - [ ] Robustness metrics in sparse regime: degradation from dense -> sparse periods

## Core Results Studies

### Study 1: End-to-End Performance vs Baselines

Question: does the new joint model outperform simpler alternatives in forecasting observed targets?

- [ ] Compare against:
  - [ ] transfer-function baseline (WW -> cases/hosp)
  - [ ] pure DL forecaster without physics roll-forward
  - [ ] physics-only or reduced-parameter mechanistic baseline
- [ ] Evaluate by horizon (1/2/4 weeks) and regime (dense vs sparse)
- [ ] Report per-target wins/losses:
  - [ ] wastewater
  - [ ] reported cases
  - [ ] hospitalizations
  - [ ] deaths

Primary deliverable: table + figure showing error reduction (%) per target/horizon/regime.

### Study 2: Sparse-Data Stress Test (missingness and dropout)

Question: how well does the architecture hold up as observations disappear?

- [ ] Simulate structured missingness during training and/or evaluation:
  - [ ] drop cases only
  - [ ] drop hosp only
  - [ ] drop both cases + hosp (WW-only supervision)
  - [ ] random missing windows vs long contiguous gaps
- [ ] Compare degradation slope against baselines
- [ ] Track whether latent trajectories remain stable and physically plausible under sparse supervision

Primary deliverable: performance-vs-missingness curves and latent stability panel.

### Study 3: Post-Cases Timeline / Real-World Transition

Question: can we train in case-rich periods and continue into post-testing years without collapse?

- [ ] Train through complete case-reporting era
- [ ] Continue training/inference through 2025 with WW-dominant supervision
- [ ] Generate latent `S, I, R, D` trajectories and observation reconstructions
- [ ] Compare inferred incidence/trends against external references where possible (published estimates, surveillance summaries)
- [ ] Quantify structural breaks at transition boundaries (pre/post reporting decline)

Primary deliverable: timeline figure (2020-2025) with observed heads + inferred latent states.

## Mechanistic and Architecture Ablations

### Study 4: Loss Component Ablations

Question: which objectives are carrying performance and identifiability?

- [ ] Train variants removing one term at a time:
  - [ ] `-L_sir`
  - [ ] `-L_ww`
  - [ ] `-L_cases`
  - [ ] `-L_hosp`
  - [ ] `-L_deaths`
- [ ] Weight sweeps around defaults (`w_sir`, head weights)
- [ ] Measure both observed-target error and latent plausibility impact

Primary deliverable: ablation heatmap (metric deltas vs default model).

### Study 5: Observation Head Design Ablations

Question: do causal delay kernels and residual context materially help?

- [ ] WW head:
  - [ ] fixed shedding kernel vs learnable kernel
  - [ ] kernel length sweep `K`
- [ ] Clinical heads:
  - [ ] delay-kernel vs direct MLP mapping
  - [ ] with/without variant conditioning
  - [ ] with/without residual context from encoder
- [ ] Deaths head:
  - [ ] death-flow-based emission vs direct latent-`I` mapping

Primary deliverable: per-head architecture choice matrix with best config.

### Study 6: Spatial and Mobility Contribution

Question: how much signal comes from mobility graph and region embeddings?

- [ ] Remove mobility edges/features
- [ ] Remove static regional embeddings
- [ ] Remove both (local-only variant)
- [ ] Evaluate gains in lead/lag capture and municipality-level forecast skill

Primary deliverable: contribution chart for graph/context components.

## Training Strategy Studies

### Study 7: Curriculum and Synthetic-to-Real Transfer

Question: is curriculum still helping after adding joint inference heads?

- [ ] No curriculum (train directly on real)
- [ ] Existing phased curriculum
- [ ] Curriculum with different synthetic:real ratios in final phase
- [ ] Measure convergence speed, final metrics, and latent stability

Primary deliverable: training efficiency + final quality tradeoff chart.

### Study 8: Minimal Real Data Requirement

Question: what is the minimum amount of real supervised data needed for acceptable performance?

- [ ] Train with increasing real-data budgets (e.g., 1, 2, 3, 6, 12 months)
- [ ] Keep synthetic pretraining fixed
- [ ] Define acceptable threshold (for example: <= X% degradation from full-data model)

Primary deliverable: data-budget curve and recommended minimum onboarding requirement.

## Credibility and Trustworthiness Studies

### Study 9: Latent-State Identifiability Probes

Question: are inferred latent states uniquely constrained or underdetermined?

- [ ] Multi-seed variability analysis for latent trajectories
- [ ] Parameter posterior spread proxies (or ensemble spread)
- [ ] Counterfactual consistency checks:
  - [ ] perturb WW amplitudes
  - [ ] perturb variant mix
  - [ ] confirm directional latent response is epidemiologically coherent

Primary deliverable: identifiability risk assessment with confidence bands.

### Study 10: Out-of-Distribution Regime Shifts

Question: does the model degrade gracefully under new variants/behavior changes?

- [ ] Hold out major transition windows (variant replacement, abrupt mobility shifts)
- [ ] Evaluate recovery time and forecast error spikes
- [ ] Compare with baseline models for brittleness

Primary deliverable: regime-shift stress report.

## Execution Order (recommended)

1. [ ] Study 0 (protocol freeze)
2. [ ] Study 1 (headline benchmark)
3. [ ] Study 2 + Study 3 (sparse and post-cases claims)
4. [ ] Study 4 + Study 5 + Study 6 (mechanistic ablations)
5. [ ] Study 7 + Study 8 + Study 9 (data and training efficiency)
6. [ ] Study 10 + Study 11 (credibility and robustness)

## Publication-Ready Result Pack

- [ ] Main table: benchmark results by horizon/target/regime
- [ ] Main figure: 2020-2025 timeline with latent and observed trajectories
- [ ] Ablation figure: loss/head/dynamics component impacts
- [ ] Robustness figure: missingness and regime-shift stress tests
- [ ] Appendix: config list, seeds, and dataset slice definitions for reproducibility

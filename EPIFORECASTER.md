# Joint Inference-Observation Epidemiological Forecaster

## 1. Overview

The `EpiForecaster` implements a **Joint Inference-Observation** framework designed for real-world epidemiological forecasting where the "True Infection Rate" is unknown.

The model infers latent epidemiological states ($S, I, H, R, D$) via a differentiable physics roll-forward and anchors them to observable proxy signals (Wastewater, Hospitalizations, Reported Cases, and Deaths).

The core philosophy is: **Physics constrains the Latent State, while Observations constrain the Physics.**

## 2. Architecture

The model operates as a three-stage differentiable pipeline:

### Stage 1: The Encoder (TransformerBackbone)
*   **Role**: Amortized inference engine. Estimates the parameters of the dynamical system, initial states, and observation context from historical data.
*   **Input**:
    *   **Clinical History**: Hospitalizations, Deaths, Reported Cases ($B \times T \times 3$ each).
    *   **Biomarkers**: Wastewater viral load features ($B \times T \times D_{bio}$).
    *   **Mobility**: Dynamic graph embeddings from `MobilityGNN` (optional).
    *   **Regional Embeddings**: Static geospatial embeddings (optional).
    *   **Population**: Static population count (optional).
    *   **Temporal Covariates**: Cyclic day-of-week (sin/cos) and holiday indicators ($B \times T \times 3$, optional).
*   **Output**:
    *   **Transmission Rate ($\beta_t$)**: Time-varying infection rate.
    *   **Recovery Rate ($\gamma_t$)**: Time-varying recovery rate.
    *   **Direct Mortality Rate ($\mu^I_t$)**: Time-varying non-hospital death rate from infected state.
    *   **Hospitalization Rate ($\alpha_t$)**: Time-varying admission rate from infected to hospitalized state.
    *   **Hospital Recovery Rate ($\rho_t$)**: Time-varying discharge/recovery rate from hospitalized state.
    *   **Hospital Mortality Rate ($\mu^H_t$)**: Time-varying severe mortality rate from hospitalized state.
    *   **Initial States**: ($S_0, I_0, H_0, R_0$) fractions summing to 1.
    *   **Observation Context**: Residual features for observation heads to model non-SIRHD dynamics (e.g., reporting delays, ascertainment noise).

### Stage 2: The Physics Core (SIRRollForward)
*   **Role**: Generates latent trajectories consistent with SIRHD dynamics.
*   **Mechanism**: A differentiable Euler integration step.
    $$ \frac{dS}{dt} = -\beta_t \frac{SI}{N} $$
    $$ \frac{dI}{dt} = \beta_t \frac{SI}{N} - \gamma_t I - \alpha_t I - \mu^I_t I $$
    $$ \frac{dH}{dt} = \alpha_t I - \rho_t H - \mu^H_t H $$
    $$ \frac{dR}{dt} = \gamma_t I + \rho_t H $$
    $$ \frac{dD}{dt} = \mu^I_t I + \mu^H_t H $$
*   **Latent Contract**: All states are modeled as **fractions** of the population ($S+I+H+R+D=1.0$). The physics core uses a normalized population $N=1.0$ internally to ensure scale invariance.
*   **Outputs**: Trajectories for $S, I, H, R, D$, `infection_flow` (flux out of $S$, $\beta_t S_t I_t / N$ of each Euler step, reported before the non-negativity clamp and mass rescale), `hospitalization_flow` (flux into $H$), and `death_flow` (flux into $D$).

### Stage 3: Observation Heads
Differentiable layers mapping latent states to observable metrics.

1.  **Wastewater Head (`pred_ww`)**:
    *   **Mechanism**: Convolves a latent source with a shedding kernel. The source is
        $I$ (prevalence, default) or, with `model.observation_heads.ww_source:
        infection_flow`, the S→I incidence, as in renewal-based wastewater models;
        the origin incidence is the first Euler step's flux. Set
        `relevant_source_ww` to the same value.
    *   **Function**: $\text{source}_{t-k:t} \to \text{Viral Load}_t$
2.  **Hospitalization Head (`pred_hosp`)**:
    *   **Mechanism**: Applies delay kernel and scaling to the latent hospitalization/admission flow.
    *   **Function**: $\text{hospitalization\_flow}_{t-k:t} \to \text{Hospital Admissions}_t$
3.  **Cases Head (`pred_cases`)**:
    *   **Mechanism**: Models ascertainment rate and reporting delays.
    *   **Function**: $I_{t-k:t} \to \text{Reported Cases}_t$
4.  **Deaths Head (`pred_deaths`)**:
    *   **Mechanism**: Maps death flow (latent mortality) to observed deaths.
    *   **Function**: $\text{death\_flow}_{t-k:t} \to \text{Observed Deaths}_t$

## 3. Loss Landscape

The model is trained end-to-end using a composite loss function:

$$
L_{\mathrm{total}}
=
\sum_{h \in A} \alpha_h L_h
+ w_{\mathrm{SIRHD}} L_{\mathrm{SIRHD}},
$$

where \(A\) is the set of observation heads with eligible targets in the
current batch. The observation weights \(\alpha_h\) sum to
`gradnorm_obs_weight_sum`.

### Probabilistic observation objectives

Each observation head emits a predictive location \(\mu\) and positive scale
\(\sigma\). Clinical targets use `log1p(per-100k)`; wastewater uses
`log1p(concentration)`. Existing wastewater metric keys retain the legacy
`log1p_per_100k` suffix. Production training uses the zero-clipped WIS on a
Normal censored at zero (`observation_loss: wis`, `censored_location: true`;
campaign `censored_likelihood_v1`). The Normal negative log likelihood is the
objective of synthetic pretraining and the former production default:

$$
L_{\mathrm{NLL}}(y,\mu,\sigma)
=
\frac{1}{2}\left(\frac{y-\mu}{\sigma}\right)^2
+ \log \sigma
+ \frac{1}{2}\log(2\pi).
$$

Normal continuous ranked probability score is also available as a training
objective:

$$
L_{\mathrm{CRPS}}(y,\mu,\sigma)
=
\sigma\left[
z\left(2\Phi(z)-1\right)
+ 2\phi(z)
- \frac{1}{\sqrt{\pi}}
\right],
\qquad z=\frac{y-\mu}{\sigma}.
$$

Both are proper scoring rules. NLL is more sensitive to tail misses and
overconfident scales; CRPS remains in target-space units and is less dominated
by individual tail events. Checkpoint selection uses the equal-target mean
validation CRPS (`mean_target_crps`) when it is available. NLL training falls
back to mean target NLL when CRPS is unavailable; the general fallback is mean
target MAE.

Fixed-df Student-t NLL and CRPS are also supported. Student-t CDF values use an
exact trigonometric recurrence for integer df up to 100 and 64-point numerical
integration otherwise. WIS training and evaluation use a fixed 23-quantile grid
for both families. These are transformed-space scores, not raw-incidence CRPS.

Research campaigns set `training.selection_metric=mean_target_wis` and declare
`training.selection_required_heads`. Explicit selection rejects a missing or
non-finite required score and invalid observed predictions; it cannot fall back
to a different metric. General training retains `selection_metric=auto`.
Canonical per-head aggregates include WIS, CRPS, NLL, interval coverage/width,
and validity counts. Granular schema 5 retains distribution parameters, WIS,
and observation support for later calibration and event evaluation.

Targets are `log1p` values and cannot be negative, so WIS and the central
intervals use every forecast quantile clipped at zero, $\max(q, 0)$. This scores
the forecast censored at zero: interval mass below zero is not counted as
width, and the point of the WIS median term is $\max(\mu, 0)$. Interval bounds,
widths and coverage in the summaries and granular rows are the clipped ones
(granular schema 5; schema 4 and earlier hold unclipped values and are not
comparable). The WIS training objective uses the same definition. NLL and CRPS
are the scores of the unclipped distribution unless the head is censored.

`model.observation_heads.censored_location: true` makes every head a Normal
censored at zero. The anchored location is no longer clamped: `dist_<head>_loc`
is the latent location and may be negative, which is how a head states a
probability of zero above one half, and `pred_<head>` is the censored median
$\max(\mu, 0)$. NLL scores an observed zero by its probability mass,
$-\log\Phi(-\mu/\sigma)$, and a positive value by the Normal density; CRPS is the
closed form for a point mass on the lower bound. Both apply to the `nll` and
`crps` training objectives and to the evaluation summaries, so their values are
not comparable with an uncensored run; compare such runs on WIS. The setting
requires the normal family. `mechanistic_<head>` and `residual_<head>` stay the
decomposition of the point prediction. Research calibration receives the
setting and scores NLL, CRPS and the point error of the censored distribution;
the PIT of an observed zero is drawn uniformly on its probability mass (seeded
randomized PIT). Its 80% coverage multiplier treats a target on the bound as
covered once the clipped lower end reaches it.

Every head's scale is a softplus plus `model.observation_heads.distribution_scale_floor`.
The floor is the only bound on the Normal density of an observed zero: an
anchored location sits exactly on the zero clamp, so a collapsed scale earns
$\log\sigma$ on each zero and pays $(y/\sigma)^2/2$ on the next count. One count
in a region of $N$ residents is $\log(1 + 10^5/N)$ in target space (3.5 at
$N=3000$, 0.7 at $N=10^5$), so
`count_step_scale_floor_<head>` (`hosp`, `cases`, `deaths`; default 0) raises the
floor to $\max(\text{floor},\, f \log(1 + 10^5/N))$. A zero forecast then misses a
single count by at most $1/f$ standard deviations in every region. The setting
assumes integer counts of the target region: in `real_with_id_v2` only cases
qualify, because hospitalizations and deaths are allocated fractions. The
`scale_floor_count_<head>` validity count still refers to the constant floor.

### Observation weighting and supervision

Per-target masks fully exclude imputed points:

* `w_t = 1.0` for observed points;
* `w_t = 0.0` for imputed or missing points.

Each head is supervised only when its history and forecast window meet the
configured observed-fraction thresholds. Fractions are evaluated against
expected reporting opportunities: daily for cases/deaths and every seven days
for hospitalizations/wastewater. The per-head score is

$$
L_h
=
\frac{\sum_t w_t\, S(y_t,\mu_t,\sigma_t)}
     {\sum_t w_t + \epsilon},
$$

where \(S\) is the configured point or probabilistic score.

With `adaptive_scheme=none`, active heads receive a fixed equal split.
GradNorm is available only with non-negative observation objectives because its
relative training-rate calculation assumes non-negative task losses. The
paired ablation registry therefore compares:

* `baseline`: zero-clipped WIS on the censored Normal with fixed weighting;
* `loss:crps`: censored Normal CRPS with fixed weighting;
* `gradnorm:crps`: censored Normal CRPS with adaptive GradNorm weighting.

Synthetic runs may additionally provide dense SIRHD compartment targets.
`training.loss.joint.latent.total_weight` controls this regularizer. The five
tasks are S, I, H, R, and destination-dated death flow. The configured latent
mode can use raw fraction-space MSE, fixed training-split standard-deviation
scaling, or fixed `asinh` scaling for I, H, and death flow. In `asinh` mode
these three are compared as $\operatorname{asinh}(x/c)$ with
$c = \texttt{asinh\_scale\_fraction} \times \sigma_{\mathrm{train}}$; a fraction
well below 1 makes the comparison logarithmic over realistic magnitudes, so a
collapsed small latent keeps a usable gradient through the initial-state
softmax and the multiplicative roll-forward. Synthetic pretraining can require
all five finite, nonnegative targets; other runs use a zero latent weight.

### Latent plausibility indicators (evaluation)

Real data has no latent targets, so evaluation emits ground-truth-free
indicators of whether the inferred SIRHD latents are epidemiologically
plausible. They use every forecast window and are computed by
`TorchLatentPlausibilityAccumulator` in every evaluation pass, including
per-epoch validation. Only two outputs write them, with prefix
`latent_plausibility_` next to the `latent_*` and `observation_latent_*`
metrics: the trainer test summary (`test_summary.json` `metrics`) and the CLI
`eval epiforecaster --summary-json` file. Quantile suffixes are `q05`,
`q25`, `median`, `q75`, `q95` (exact `torch.quantile` across windows; NaN and
non-finite windows excluded). All quantities are population fractions ($N=1$).

| Key | Definition |
| --- | --- |
| `window_count` | Forecast windows evaluated |
| `i_origin_<q>` | $I$ at the forecast origin (`I_trajectory[:, 0]`) |
| `i_peak_<q>` | $\max_h I$ over the trajectory |
| `i_collapse_fraction` | Fraction of windows with $I$ peak $< 10^{-6}$ (0.1 per 100k) |
| `s_depletion_origin_<q>` | $1 - S_0$ |
| `forecast_attack_<q>` | $S_0 - S_H$, fraction infected over the forecast |
| `rt_origin_<q>` | $\beta_0 S_0 / (\gamma_0 + \alpha_0 + \mu^I_0)$ |
| `rt_horizon_mean_<q>` | Per-window mean over $h=0..H-1$ of $\beta_h S_h / (\gamma_h + \alpha_h + \mu^I_h)$ |
| `ascertainment_<head>_<q>` | Per-window implied ascertainment, `head` in `cases`, `hosp`, `deaths` |
| `ascertainment_<head>_pooled` | Total observed per-100k over total latent per-100k |
| `ascertainment_<head>_window_count` | Windows entering the ascertainment quantiles |
| `ascertainment_<head>_zero_latent_window_fraction` | Share of observed windows whose latent flow sums to zero |

$\alpha$ is the $I \to H$ rate (`hospitalization_rate_t`) and $\mu^I$ is the
direct $I \to D$ mortality (`mortality_t`); hospital mortality is not part of
the removal rate from $I$. Steps with a non-positive denominator or non-finite
value are skipped, and a window is excluded only when no step is valid.

Implied ascertainment compares observed counts with the latent flow feeding the
head: `infection_flow` for cases, `hospitalization_flow` for hospitalizations,
and `death_flow` for deaths. Supervised targets are absolute
`log1p(per-100k)` (`delta_forecasting` only anchors predictions), so the
observed count per 100k is `expm1(target)` and the latent count per 100k is the
flow times $10^5$, aligned to the forecast targets. Per window the ratio is
$\sum_t w_t\, y_t / \sum_t w_t\, \ell_t$ over supervised days ($w_t>0$); windows
need $\sum_t w_t > 0$ and a positive latent sum to enter the quantiles. The
pooled ratio sums over all windows with supervision and finite sums, including
zero-latent windows. In the real dataset, cases and deaths are daily counts and
hospitalizations are weekly totals placed on the week-start date
(`data/preprocess/processors/hospitalizations_processor.py`), so the `hosp`
ratio divides a weekly total by one day's latent flow and is about seven times
an admission ascertainment. Synthetic corpora follow their generator's
reporting layer. The a97c812 corpus (reporting v2) passes counts through a
magnitude calibration in `log1p(per 100k)` space that puts observed deaths about
8-10 times above the latent death flow, so its synthetic `deaths` ratio is not
an ascertainment.

### Latent information indicators (evaluation)

Plausibility indicators check latent levels; the information indicators check
whether the latent feeding each head tracks the observed series and drives the
forecast. `TorchLatentInformationAccumulator`
(`evaluation/latent_information.py`) runs in final test evaluations (trainer
test split and `eval_checkpoint`, not per-epoch validation) and writes
`latent_info_<head>_*` keys, `head` in `hosp`, `ww`, `cases`, `deaths`, to
`test_summary.json` `metrics` and the CLI `--summary-json` file.

The `training.latent_information` block controls it: `enabled` (default `true`;
`false` omits every `latent_info_*` key), `resample_count` (default `200`
permutations and bootstrap resamples behind the `_null95`, `_q05` and `_q95`
bounds) and `max_resample_windows` (default `20000`; above it a seeded
subsample enters the resamples). The indicators move every batch to CPU, so
`enabled: false` also removes that cost from a test evaluation.

Each series is centred within its forecast window over supervised days, so a
region's level and the delta-forecasting anchor carry no information. The latent
is the head's `relevant_source` compared as $\log(1 + 10^5 x)$.

|Key|Definition|
|---|---|
|`r2_lat_obs`, `r2_lat_pred`|Squared same-day correlation of latent with observation / forecast location|
|`mi_lat_obs`, `mi_lat_pred`, `mi_pred_obs`|Gaussian-copula mutual information, bits|
|`te_lat_obs`, `te_pred_obs`|Linear-Gaussian transfer entropy to the observation one reporting step ahead (1 day cases/deaths, 7 days hosp/ww), bits|
|`sign_lat_obs`, `sign_lat_pred`|Fraction of step changes with the same sign|
|`resid_share`|Within-window variance of the residual contribution over that of the forecast|
|`r2_lat_obs_weekly`, `resid_share_weekly`|Cases and deaths: the same on 7-day block means, which removes the day-of-week reporting cycle|
|`lag_r2_lat_obs_<m\|p><d>`, `best_lag_lat_obs`, `best_lag_r2_lat_obs`|Lag scan; `p` means the latent leads (±14 d daily heads, ±21 d in 7-day steps weekly heads)|
|`<metric>_week_total`|Hospitalizations with `relevant_source_hosp: hospitalization_flow`: the four latent-observation metrics and `r2_lat_pred` with the latent summed over days $h..h+6$, the days a weekly total dated on its week start covers. Weeks leaving the horizon are dropped (`week_total_window_count`, `week_total_observation_count`)|
|`r2_true_obs`, `mi_true_obs`, `te_true_obs`, `sign_true_obs`, `r2_true_obs_weekly`, `lag_r2_true_obs_<m\|p><d>`, `best_lag_true_obs`, `best_lag_r2_true_obs`|Synthetic corpora: the latent-observation metrics with the stored true latent in place of the inferred one, the ceiling of an exactly recovered latent (`true_window_count`, `true_observation_count`)|
|`r2_lat_true`|Synthetic corpora: within-window squared correlation of the inferred and the true latent|
|`<metric>_null95`|95th percentile over `resample_count` permutations pairing each window's observations with another window's latent (latent-observation metrics)|
|`<metric>_q05`, `<metric>_q95`|Region-bootstrap interval, `resample_count` resamples; region sampling only|

Permutations and bootstraps use at most `max_resample_windows` seeded windows
(`resample_window_count`). Overlapping windows make the effective sample smaller
than `observation_count`.

True latents exist for the sources `I`, `H`, `D` and `death_flow`. The corpus
stores no true admission or infection flow, so the default hospitalization
source has no ceiling; evaluating with
`model.observation_heads.relevant_source_hosp=H` gives the ceiling of the
hospital census instead.

## 4. Model Variants

The architecture supports ablation studies via `ModelVariant` flags:

*   **Regions**: Enables static `region_embeddings` inputs.
*   **Mobility**: Enables `MobilityDenseEncoder` to process dynamic mobility graphs.
*   **Biomarkers**: Enables wastewater inputs.
*   **Cases**: Enables clinical history inputs.

`model.observation_heads.mechanistic_disabled_heads` sets a head's mechanistic
log location to an exact zero, so it predicts from the observation-context
residual alone. Disabling every head gives a black-box control with the same
encoder; a relative residual budget (`training.residual_schedule`) is rejected
on such heads, and a head cannot disable both paths.

### Mechanistic integration variants

All default to the behaviour described above. With the defaults the
roll-forward starts at the forecast origin, so each kernel sees only the
forecast-period latent and the origin admission flow is zero.

|Setting|Effect|
|---|---|
|`model.sir_physics.rollforward_history_steps: L`|The initial state is placed $L$ days before the origin and rolled through the history. Forecast-step rates still come from the final hidden state; the rate of history step $k$ comes from a per-day rate head on the encoding of day $k$, initialized stationary ($\beta$ equal to the removal-rate prior). $L$ must be at least the longest observation kernel and below `input_window_length`. Forecast outputs keep their shapes; `observation_source_<head>` covers the whole roll-forward.|
|`training.loss.joint.history_reconstruction_weight`|Adds the mean over heads of the masked MSE between `mechanistic_history_<head>` and the observed history, in `log1p` space. The history location is the unanchored mechanistic location without residual, from the first day with a full kernel window to the last history day. Requires $L>0$.|
|`model.sir_physics.time_varying_rates: beta`|$\gamma$, $\mu^I$, $\alpha$, $\rho$ and $\mu^H$ are the horizon mean of their rate head, constant over the window. With `rollforward_history_steps: 0` checkpoints are interchangeable with `all`; with $L>0$ they are not, because `history_rate_projection` has one output instead of six.|
|`training.loss.joint.beta_smoothness_weight`|Adds the mean squared first difference of $\log\beta$ along the roll-forward (random-walk prior).|
|`training.loss.joint.residual_penalty_weight`|Adds the mean over heads of the mean squared residual contribution to the forecast location over supervised points. Rejected with `mechanistic_disabled_heads`.|
|`model.observation_heads.hosp_week_total: true`|The hospitalization head predicts the admission flow summed over days $h..h+6$, the days a weekly total dated on its week start covers. The roll-forward runs six days past the horizon with the last rates held.|
|`model.sir_physics.force_of_infection: renewal`|The $S \to I$ flux is $R_t S_t \sum_{s=1}^{G} g_s\, i_{t-s}$ with $R_t = \beta_t/(\gamma_t+\alpha_t+\mu^I_t)$ and $g$ a discretized Gamma generation interval (`generation_interval_mean`, `generation_interval_sd`, `generation_interval_length`), capped at $S_t$. Days before the roll-forward are seeded with $(\gamma_0+\alpha_0+\mu^I_0) I_0$, so the first flux equals mass action. Compartment bookkeeping is unchanged. The `latent_diag_teacher_forced_one_step_*` diagnostic is omitted, because each one-step call would be re-seeded and score mass action; `latent_diag_oracle_x0_*` is still reported.|
|`model.observation_heads.cases_source: infection_flow`|The cases delay kernel convolves incidence instead of prevalence. Set `relevant_source_cases` to the same value.|

Evaluation reports the unweighted regularizers as `loss_history_reconstruction`,
`loss_residual_penalty` and `loss_beta_smoothness` for every run.

## 5. Configuration Parameters

### Temporal Covariates (`model.params.include_day_of_week`, `model.params.include_holidays`)

Controls whether day-of-week and holiday features are fed into the encoder:

| Flag | Default | Description |
|------|---------|-------------|
| `include_day_of_week` | `false` | Adds `dow_sin`, `dow_cos` (2 features) |
| `include_holidays` | `false` | Adds `is_holiday` (1 feature) |

When both flags are `false` (default), the model auto-detects temporal covariates from the dataset.

**Configuration example**:
```yaml
model:
  params:
    include_day_of_week: true
    include_holidays: true
```

**Prerequisites**: When enabled, the dataset must include a `temporal_covariates` tensor. Configure this in preprocessing:

```yaml
temporal_covariates:
  include_day_of_week: true
  include_holidays: true
  holiday_calendar_file: data/files/catalonia_holidays.csv
```

The holiday calendar should be a CSV with a `date` column containing holiday dates (e.g., `2020-01-01`).

### Mobility graph processing

Canonical mobility values are stored as `log1p(raw_flow)`, but edge selection is
configured in raw-flow units:

```yaml
data:
  mobility_min_flow_raw: 10.0
  mobility_weight_clip_max: 8.0
```

The dataset converts the raw threshold with `log1p`, selects finite positive
non-self edges before transforming weights, and keeps retained weights in
nonnegative log space with upper clipping. The resulting dynamic edge mask is
shared by dense GNN adjacency, k-hop reachability, clinical context, imported
risk weights, and GPU adjacency injection. Self-loops are added separately.

For `spatial_queen` and `spatial_knn`, the selected static adjacency controls
GNN weights, k-hop reachability, and clinical context. Imported risk remains
mobility-derived.

Generate machine-readable QC artifacts and plots with:

```bash
uv run python scripts/analysis/analyze_mobility_processing.py \
  --config configs/production_only/train_epifor_mn5_full.yaml \
  --output-dir outputs/reports/mobility_processing_qc
uv run python -m dataviz.mobility_processing_qc \
  --input-dir outputs/reports/mobility_processing_qc
```

### Representative cross-validation splits

Production cross-validation uses the persisted, exactly-once 10-fold membership
in `configs/crossval/representative_10fold_pop3000.json` (280 targets of
`real_with_id_v2`, see below), which extends the 161-target
`configs/crossval/representative_10fold_mobility_refined.json`. The manifest
fixes fold identity independently of `training.seed`; `training.split_seed`
only controls generated non-manifest splits. Manifest loading rejects version,
fold-count, run-ID, dataset-region-hash, duplicate, missing, unknown, and
unbalanced-fold mismatches.

The checked-in folds start from the population, clinical coverage, geography,
queen-degree, and mobility-mass-balanced representative split. Fifteen
deterministic pair swaps reduce test-internal flow over the complete directed
mobility network, penalize high-flow events, preserve the original balance
envelope, and keep every target above 75% incoming-flow retention. No top-K
filtering is used.

Reproduce the machine-readable split analysis and plot with:

```bash
uv run python scripts/analysis/analyze_split_mobility_cut.py \
  --dataset data/processed/real_with_id_v2.zarr.zip \
  --config configs/production_only/train_epifor_mn5_full.yaml \
  --full-coverage-manifest path/to/representative_10fold_manifest.json \
  --output-dir outputs/reports/split_mobility_cut
uv run python -m dataviz.split_mobility_cut \
  --metrics outputs/reports/split_mobility_cut/split_mobility_metrics.csv \
  --summary outputs/reports/split_mobility_cut/strategy_summary.csv \
  --optimization-history outputs/reports/split_mobility_cut/optimization_history.csv \
  --balance-audit outputs/reports/split_mobility_cut/balance_audit.csv \
  --output outputs/reports/split_mobility_cut/split_mobility_cut.png
```

### Population-floor targets and region sets

`real_with_id_v2` selects target regions by a static population floor
(`valid_target_min_population: 3000`, 280 municipalities) instead of case
density, so target membership does not depend on outcomes. Its manifest,
`configs/crossval/representative_10fold_pop3000.json`, is built by
`scripts/analysis/extend_crossval_manifest.py`: the 161 former targets keep
their representative folds, and each added region joins the fold of the
former targets that share its wastewater plant (comarca mates second), with
balanced fold sizes. `diagnostics.extension.cross_fold_pairs` counts the twin
pairs that remain split across folds.

A manifest may declare `region_sets`, named static subsets of its regions
(`base`, `added`, and population bins `pop_lt_5k`, `pop_5k_20k`,
`pop_20k_100k`, `pop_ge_100k`). Test evaluation adds
`region_set_<name>_*` probabilistic metrics and a
`region_set_<name>_mean_target_wis` endpoint for each set (NaN when a head has
no support in that set), and the aggregate table carries them per head.
`training.crossval_region_set: <name>` limits training and validation regions
to one set while every fold region stays in the test role, so models trained
on different populations are scored on identical held-out support. Campaign
analysis reports paired run-level and equal-region deltas per region set
(`paired_fold_seed_region_set_deltas.csv`, `region_set_summary.csv`).

## 6. Input/Output Signature

**Forward Pass (`forward`)**:

```python
def forward(
    self,
    hosp_hist: Tensor,           # [B, T, 3]
    deaths_hist: Tensor,         # [B, T, 3]
    cases_hist: Tensor,          # [B, T, 3]
    biomarkers_hist: Tensor,     # [B, T, D_bio]
    mob_graphs: Batch,           # PyG Batch (B*T graphs)
    target_nodes: Tensor,        # [B]
    region_embeddings: Tensor,   # [num_regions, region_dim] (optional)
    population: Tensor,          # [B] (optional)
    temporal_covariates: Tensor, # [B, T, 3] (optional)
) -> dict[str, Tensor]:
```

**Returns**:

*   `beta_t`, `S_trajectory`, `I_trajectory`, `R_trajectory`: Latent dynamics; `infection_flow`, `hospitalization_flow`, `death_flow`: per-step latent flows.
*   `pred_ww`, `pred_hosp`, `pred_cases`, `pred_deaths`: Observation predictions.
*   `physics_residual`: Deviation from pure SIR dynamics (for $L_{SIR}$).

## 7. Initialization Policy (Conservative)

The model uses a conservative startup policy to keep early training dynamics stable:

*   **Prior-biased SIR heads**: Final layers for $\beta_t$, $\gamma_t$, and $\mu_t$ use small Xavier weights plus inverse-softplus prior biases (`0.25`, `0.14`, `0.002`), keeping outputs near conservative priors while preserving immediate gradient flow through projection stems.
*   **Initial state prior**: The initial-state logits head uses small Xavier weights plus $\log([0.995, 0.004, 0.001])$ bias, so `softmax` starts near a susceptible-dominant population without a step-1 gradient blind spot.
*   **Neutral observation residual path**: Observation residual projections are initialized to exact zeros, so residual corrections start inactive and become data-driven during training.
*   **Population feature stabilization**: Population inputs are transformed with `log1p(population)` before concatenation into the backbone input to avoid projection instability from large raw counts.

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
*   **Outputs**: Trajectories for $S, I, H, R, D$, `hospitalization_flow` (flux into $H$), and `death_flow` (flux into $D$).

### Stage 3: Observation Heads
Differentiable layers mapping latent states to observable metrics.

1.  **Wastewater Head (`pred_ww`)**:
    *   **Mechanism**: Convolves latent $I$ trajectory with a shedding kernel.
    *   **Function**: $I_{t-k:t} \to \text{Viral Load}_t$
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
\(\sigma\) in `log1p(per-100k)` target space. Production training uses Normal
negative log likelihood:

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

CRPS is currently implemented only for the Normal family. Scores are evaluated
in `log1p(per-100k)` space and therefore are not raw-incidence CRPS.

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

* `baseline`: Normal NLL with fixed weighting;
* `loss:crps`: Normal CRPS with fixed weighting;
* `gradnorm:crps`: Normal CRPS with adaptive GradNorm weighting.

Synthetic runs may additionally provide dense SIRHD compartment targets.
`w_sird_supervision` controls this regularizer and it automatically contributes
zero when latent targets are absent.

## 4. Model Variants

The architecture supports ablation studies via `ModelVariant` flags:

*   **Regions**: Enables static `region_embeddings` inputs.
*   **Mobility**: Enables `MobilityDenseEncoder` to process dynamic mobility graphs.
*   **Biomarkers**: Enables wastewater inputs.
*   **Cases**: Enables clinical history inputs.

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
in
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
  --dataset data/processed/real_with_id.zarr \
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

*   `beta_t`, `S_trajectory`, `I_trajectory`, `R_trajectory`: Latent dynamics.
*   `pred_ww`, `pred_hosp`, `pred_cases`, `pred_deaths`: Observation predictions.
*   `physics_residual`: Deviation from pure SIR dynamics (for $L_{SIR}$).

## 7. Initialization Policy (Conservative)

The model uses a conservative startup policy to keep early training dynamics stable:

*   **Prior-biased SIR heads**: Final layers for $\beta_t$, $\gamma_t$, and $\mu_t$ use small Xavier weights plus inverse-softplus prior biases (`0.25`, `0.14`, `0.002`), keeping outputs near conservative priors while preserving immediate gradient flow through projection stems.
*   **Initial state prior**: The initial-state logits head uses small Xavier weights plus $\log([0.995, 0.004, 0.001])$ bias, so `softmax` starts near a susceptible-dominant population without a step-1 gradient blind spot.
*   **Neutral observation residual path**: Observation residual projections are initialized to exact zeros, so residual corrections start inactive and become data-driven during training.
*   **Population feature stabilization**: Population inputs are transformed with `log1p(population)` before concatenation into the backbone input to avoid projection instability from large raw counts.

# Joint Inference-Observation Epidemiological Forecaster

## 1. Overview

The `EpiForecaster` implements a **Joint Inference-Observation** framework designed for real-world epidemiological forecasting where the "True Infection Rate" is unknown.

Instead of training on synthetic ground truth, the model infers latent epidemiological states ($S, I, R, D$) via a differentiable physics roll-forward and anchors them to observable proxy signals (Wastewater, Hospitalizations, Reported Cases, and Deaths).

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
    *   **Mortality Rate ($\mu_t$)**: Time-varying mortality rate.
    *   **Initial States**: ($S_0, I_0, R_0$) fractions summing to 1.
    *   **Observation Context**: Residual features for observation heads to model non-SIRD dynamics (e.g., reporting delays, ascertainment noise).

### Stage 2: The Physics Core (SIRRollForward)
*   **Role**: Generates latent trajectories consistent with SIRD dynamics.
*   **Mechanism**: A differentiable Euler integration step.
    $$ \frac{dS}{dt} = -\beta_t \frac{SI}{N} $$
    $$ \frac{dI}{dt} = \beta_t \frac{SI}{N} - \gamma_t I - \mu_t I $$
    $$ \frac{dR}{dt} = \gamma_t I $$
    $$ \frac{dD}{dt} = \mu_t I $$
*   **Latent Contract**: All states are modeled as **fractions** of the population ($S+I+R+D=1.0$). The physics core uses a normalized population $N=1.0$ internally to ensure scale invariance.
*   **Outputs**: Trajectories for $S, I, R$ and `death_flow` (flux into $D$ state).

### Stage 3: Observation Heads
Differentiable layers mapping latent states to observable metrics.

1.  **Wastewater Head (`pred_ww`)**:
    *   **Mechanism**: Convolves latent $I$ trajectory with a shedding kernel.
    *   **Function**: $I_{t-k:t} \to \text{Viral Load}_t$
2.  **Hospitalization Head (`pred_hosp`)**:
    *   **Mechanism**: Applies delay kernel and scaling to latent $I$.
    *   **Function**: $I_{t-k:t} \to \text{Hospital Admissions}_t$
3.  **Cases Head (`pred_cases`)**:
    *   **Mechanism**: Models ascertainment rate and reporting delays.
    *   **Function**: $I_{t-k:t} \to \text{Reported Cases}_t$
4.  **Deaths Head (`pred_deaths`)**:
    *   **Mechanism**: Maps death flow (latent mortality) to observed deaths.
    *   **Function**: $\text{death\_flow}_{t-k:t} \to \text{Observed Deaths}_t$

## 3. Loss Landscape

The model is trained end-to-end using a composite loss function:

$$ L_{total} = L_{obs} + w_{sir} L_{SIR} + w_{continuity} L_{continuity} $$

where:

*   **Training observation term (`L_obs`)**: adaptive GradNorm weighting over
    `L_WW`, `L_Hosp`, `L_Cases`, `L_Deaths` (or fixed equal split when
    `adaptive_scheme=none`).
    GradNorm uses the shared observation context probe (`obs_context`) and updates
    controller weights periodically (`gradnorm_update_every`, default 16) with EMA
    smoothing (`gradnorm_ema_decay`, default 0.9) to preserve training throughput.
*   **Evaluation observation term**: fixed equal-split weighting over active
    observation targets with sum `gradnorm_obs_weight_sum` for stable
    early-stopping comparisons.

*   **Observation Losses ($L_{WW}, L_{Hosp}, \dots$)**: Minimize error between predictions and ground truth data. Per-target masks allow training even when some signals are missing (e.g., missing wastewater data).
*   **SIR Physics Loss ($L_{SIR}$)**: A regularization term that enforces consistency between the unconstrained updates and the strict SIRD equations, preventing the model from "breaking physics" to fit noise.
*   **Continuity Loss ($L_{continuity}$)**: Optional nowcast continuity penalty that
    discourages discontinuity between the last observed point and first forecast step.

### Observation Supervision and `n_eff`

Observation heads use weighted masked reductions with three distinct controls:

1.  **Hard eligibility (`missing_permit` -> `min_observed`)**: windows with too few observed
    points for a head are excluded from supervision for that head.
2.  **Per-head weighted mean reduction**:
    $$ L_h = \frac{\sum_t w_t ( \hat{y}_t - y_t )^2}{\sum_t w_t + \epsilon} $$
3.  **Optional confidence scaling by effective support (`n_eff`)**:
    $$ n_{eff,h} = \sum_t w_t,\quad
    s_h = \min(1, n_{eff,h}/r_h)^{p},\quad
    L_h^{scaled} = s_h \cdot L_h $$
    where `p = obs_n_eff_power`, `r_h` is head-specific reference
    (`*_n_eff_reference`, then `obs_n_eff_reference`, then 1.0 fallback).

Current production policy is to **fully mask imputed points out of loss**:

*   `w_t = 1.0` for observed points
*   `w_t = 0.0` for imputed/missing points

With this policy, `n_eff` is effectively the count of truly observed supervised points
(after gating), and the `power` term controls how aggressively low-support batches are
down-weighted:

*   `p = 0`: disabled
*   `p = 1`: linear count-based scaling
*   `0 < p < 1`: gentler than linear
*   `p > 1`: harsher than linear

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

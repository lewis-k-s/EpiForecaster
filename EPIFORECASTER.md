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

$$ L_{total} = w_{ww} L_{WW} + w_{hosp} L_{Hosp} + w_{cases} L_{Cases} + w_{deaths} L_{Deaths} + w_{sir} L_{SIR} $$

*   **Observation Losses ($L_{WW}, L_{Hosp}, \dots$)**: Minimize error between predictions and ground truth data. Per-target masks allow training even when some signals are missing (e.g., missing wastewater data).
*   **SIR Physics Loss ($L_{SIR}$)**: A regularization term that enforces consistency between the unconstrained updates and the strict SIRD equations, preventing the model from "breaking physics" to fit noise.

## 4. Model Variants

The architecture supports ablation studies via `ModelVariant` flags:

*   **Regions**: Enables static `region_embeddings` inputs.
*   **Mobility**: Enables `MobilityPyGEncoder` to process dynamic mobility graphs.
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

*   **Prior-biased SIR heads**: Final layers for $\beta_t$, $\gamma_t$, and $\mu_t$ are initialized with zero weights and biases set via inverse-softplus to epidemiologically plausible priors (`0.25`, `0.14`, `0.002`).
*   **Initial state prior**: The initial-state logits head starts from $\log([0.995, 0.004, 0.001])$, so `softmax` begins near a susceptible-dominant population.
*   **Neutral observation residual path**: Observation residual projections are initialized to exact zeros, so residual corrections start inactive and become data-driven during training.
*   **Population feature stabilization**: Population inputs are transformed with `log1p(population)` before concatenation into the backbone input to avoid projection instability from large raw counts.

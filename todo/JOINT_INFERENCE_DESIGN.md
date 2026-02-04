# Joint Inference-Observation Framework Design

**Status:** Draft
**Context:** Transition from synthetic supervision to real-world "hidden state" inference.

## 1. Overview

This document details the architectural transition from a purely supervised learning paradigm (where $I_{true}$ is known) to a **Joint Inference-Observation** framework. In the real-world post-testing regime (2022-2026), the "True Infection Rate" is unknown. We infer latent epidemiological states ($S, I, R$) via a physics roll-forward and anchor them to observable proxy signals (Wastewater, Hospitalizations) with differentiable observation heads. Observation heads may receive residual context from mobility/embeddings to preserve signal beyond the physics core.

## 2. The Extended Architecture

The model operates as a three-stage pipeline. The "Magic" is that the middle stage (Latent SIR) is constrained by physics, forcing the first stage (Encoder) to produce biologically meaningful results to satisfy the third stage (Observations).

### Stage 1: The Encoder (GNN-Transformer)
* **Role:** Estimates the parameters of the dynamical system, initial states, and observation context.
* **Input:** 
  * Multi-municipality Mobility (Dynamic graphs)
  * Raw Sarsaigua Gene Counts (Wastewater)
  * Variant Proportions (Freyja 2 deconvolution)
  * Static Regional Embeddings
* **Output:** 
  * Transmission Rate ($\beta_t$)
  * (Optional) Recovery Rate ($\gamma$) if not fixed.
  * Initial states ($S_0, I_0, R_0$)
  * Observation context/residual features for emission heads

### Stage 2: The Physics Core (SIR Roll-Forward)
* **Role:** A differentiable roll-forward that generates latent trajectories.
* **Mechanism:** 
  * It computes the *expected* derivatives based on the SIR differential equations:
    $$ \frac{dS}{dt} = -\beta \frac{SI}{N}, \quad \frac{dI}{dt} = \beta \frac{SI}{N} - \gamma I, \quad \frac{dR}{dt} = \gamma I $$
  * It rolls forward from $(S_0, I_0, R_0)$ with $\beta_t$ (and $\gamma$), producing $(\hat{S}, \hat{I}, \hat{R})$ trajectories.
  * It can also provide a residual loss as a soft constraint if we introduce optional latent residuals later.

### Stage 3: The Observation Heads (Emission Models)
These are differentiable layers that map the latent $I$ state to observable metrics. Heads may consume residual context features from the encoder (mobility/embeddings) to improve mapping without violating physics.

**Parameterization note:** learnable kernel weights and observation/scaling rates are constrained to be positive (e.g., softplus/exp) and kernels are normalized to sum to 1 for probabilistic interpretation.

1.  **Wastewater Head (Shedding Convolution)**
    *   **Mechanism:** A 1D-Convolution with a fixed or learnable kernel representing the viral shedding profile (fecal shedding dynamics).
    *   **Mapping:** $I_{t-k:t} \to \text{Estimated Viral Load}_t$
    *   **Variant Awareness:** Can adjust shedding intensity based on dominant variants.

2.  **Clinical Head (Hospitalization)**
    *   **Mechanism:** An MLP or Delay-Kernel.
    *   **Mapping:** $I_{t-k:t} \to \text{Estimated Hospitalizations}_t$
    *   **Variant-Specific Risk:** Uses variant proportions to learn that Variant A might have a higher hospitalization rate ($\rho_A$) than Variant B ($\rho_B$).
        $$ \text{Hosp}_t \approx \sum_{v \in Variants} (I_t \cdot \text{Prop}_{v,t} \cdot \rho_v) * \text{DelayKernel} $$

## 3. Loss Landscape

Since "True $I$" is unknown, we train "End-to-End" using the **Total Loss**:

$$ L_{total} = w_1 L_{WW} + w_2 L_{Hosp} + w_3 L_{SIR} $$

1.  **Wastewater Loss ($L_{WW}$):**
    *   High-frequency signal.
    *   Anchors the **timing** of the infection curves.
    *   Objective: Minimize distance between *Predicted Viral Load* and *Observed Gene Counts*.

2.  **Hospitalization Loss ($L_{Hosp}$):**
    *   Low-frequency, high-confidence signal.
    *   Anchors the **scale** of the infection.
    *   Objective: Minimize distance between *Predicted Admissions* and *Observed Admissions*.

3.  **SIR Loss ($L_{SIR}$):**
    *   The "Glue" / Regularizer.
    *   Forces the latent $\hat{I}$ to follow smooth SIR dynamics rather than overfitting to noisy wastewater spikes.
    *   Objective: $$ || \Delta \hat{I}_t - (\beta_t \frac{\hat{S}_t \hat{I}_t}{N} - \gamma \hat{I}_t) ||^2 $$

## 4. Addressing the Catalan Context (2022-2026)

This architecture specifically solves the challenges of the post-acute phase:

*   **The "Reporting Gap":** Even if clinical testing vanishes (no Case signal), the model infers $I$ because it must satisfy the $L_{SIR}$ physics while fitting $L_{WW}$. The $L_{SIR}$ term bridges gaps in data.
*   **Variant Shifts:** By explicitly feeding variant proportions into the Clinical Head, the model learns disjoint risks. If a new variant spikes in Wastewater but Hospitalizations stay low, the model learns a lower $\rho_{new}$ rather than suppressing the estimated $I$.

## 5. Differentiation from Existing Tools

Unlike standard "Transfer Functions" (linear correlations) used in current dashboards (e.g., Cetaqua/ICRA):

1.  **Non-Linear Spatial Dynamics:** The GNN captures how outbreaks flow between municipalities (e.g., *L'Hospitalet* $\to$ *Barcelona*) via mobility, rather than treating each point in isolation.
2.  **Biological Constraints:** The SIR Roll-Forward prevents "unphysical" predictions (like instantaneous 50% drops) that pure DL models might hallucinate, ensuring policy-relevant stability.

## 6. Implementation Contract (Pre-Implementation)

Lock these details before coding to avoid churn and mismatched interfaces:

1.  **Latent Parameterization + Roll-Forward Contract**
    *   Backbone outputs: $\beta_t$, optional $\gamma$, $S_0/I_0/R_0$, and observation context.
    *   Roll-forward equations, timestep convention (daily/weekly), and bounds (nonnegativity, mass conservation).

2.  **Observation Heads + Delay Kernel Spec**
    *   Causal kernel length $K$, positivity/normalization constraints, and edge handling.
    *   Head inputs: $I_t$ only vs $I_t$ + context/variants; expected tensor shapes.

3.  **Loss Composition + Scaling**
    *   Explicit loss formula, weights, and normalization (per-capita, log, or z-scored).
    *   Missing targets: mask vs disable head; zero-weight behavior in total loss.

4.  **Data Requirements + Dataset Schema**
    *   Required inputs (population $N$, variants, mobility, wastewater, hosp/deaths).
    *   Defaults and config toggles when fields are absent.

5.  **Forward Signature + Return Schema**
    *   Exact output dict (latents, params, obs preds, context).
    *   Training vs inference differences and which outputs are used for losses/metrics.

## 7. Implementation Checklist

- [x] **Define `ObservationHead` Class:**
    - [x] `SheddingConvolution` layer.
    - [x] `HospitalizationDelay` layer.
- [x] **Define `SIRRollForward` Class:**
    - [x] Differentiable SIR step function.
    - [x] Roll-forward logic for $(\hat{S}, \hat{I}, \hat{R})$.
- [ ] **Update `EpiForecaster` Model:**
    - [ ] Integrate roll-forward SIR and heads into the forward pass.
    - [ ] Return dictionary of `{latent_I, pred_ww, pred_hosp, physics_residuals, obs_context}`.
- [ ] **Update Trainer:**
    - [ ] Implement the composite loss function.
    - [ ] Add weighting hyperparameters ($w_1, w_2, w_3$) to Config.

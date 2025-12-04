# Epidemiological Forecasting Pipeline with Mobility GNN and Attention

## 1. Overview

This document specifies the design of a forecasting pipeline for epidemiological time series that integrates:

* Case incidence time series
* Wastewater biomarker time series
* Static regional (geospatial) embeddings
* Dynamic mobility-based graph neural network (GNN) features
* A Transformer-based forecasting head using temporal self-attention

The primary task is to forecast future **case incidence** at the level of geospatial regions. The design supports multiple model variants to ablate the contributions of regional embeddings and mobility.

Key design commitment:

* **MobilityGNN is non-recurrent over time.** It operates on each time slice independently, using cases and origin–destination mobility at that time step.
* **Temporal dynamics** are handled entirely by the **ForecasterHead**, which is a Transformer that applies self-attention over the historical sequence of per-time-step features.

---

## 2. Problem Formulation

We consider a discrete time index:

* Time steps: $t \in \{1, \dots, T\}$
* Forecasting horizon: $H$ (number of future time steps to predict)
* Historical context length: $L$ (number of past time steps used as input)

For each **region** $i \in \{1, \dots, N\}$ in the **case layer** C, we have:

* Case incidence series $y_{i,t}$ (e.g., cases per 100k population)
* Biomarker series $b_{i,t,f}$ for biomarkers $f \in \{1, \dots, F\}$

Wastewater biomarker measurements originate at catchment areas (layer W) and are mapped to case regions (layer C) via a static contribution matrix.

The forecasting objective is, for each region $i$:

$$
\hat{y}_{i,t+1:t+H} = f_\theta(\text{history of cases, biomarkers, mobility, geospatial context})
$$

The model learns parameters $\theta$ that minimize a forecasting loss between $\hat{y}_{i,t+1:t+H}$ and observed $y_{i,t+1:t+H}$.

---

## 3. Data Model and Preprocessing

### 3.1 Entities and Layers

* **Case regions (layer C)**

  * Number of patches: $N$
  * Represent, for example, municipalities or districts.

* **Wastewater catchment areas (layer W)**

  * Number of patches: $M$
  * Each catchment serves one or more case regions.

* **Biomarkers**

  * Number of biomarkers: $F$
  * Examples: pathogen-specific RNA concentrations, surrogate markers, etc.

* **Mobility**

  * Origin–destination flows between regions in C at each time step.

* **Static regional attributes**

  * Used to compute region embeddings through a GNN over a static geography graph.

### 3.2 Raw Data Dimensions

We denote batches where useful, but core dimensions are below.

* **Epidemic series**:

  * Cases at region level:

    * `cases` ($\in \mathbb{R}^{N \times T}$)
  * Biomarkers at catchment level (raw):

    * `biomarkers_raw` ($\in \mathbb{R}^{M \times T \times F}$)

* **Mobility** (origin–destination):

  * `mobility` ($\in \mathbb{R}^{T \times N \times N}$)
  * At time $t$, `mobility[t, j, i]` is the flow from origin region $j$ to destination region $i$.

* **Regions and static features**:

  * `region_features` ($\in \mathbb{R}^{N \times D_{\text{static}}}$)

    * Demographic, geographic, infrastructure, etc.
  * Static adjacency / graph for regions C (e.g., contiguity or distance-based).

* **Dimension mapper (catchment → regions)**:

  * `relation` ($\in \mathbb{R}^{M \times N}$)

    * `relation[m, n]` is the contribution ratio from catchment $m$ to region $n$.

### 3.3 Wastewater Aggregation to Regions

To aggregate catchment-level biomarkers to region-level:

* Let $R \in \mathbb{R}^{M \times N}$ be the normalized contribution matrix.
* Raw biomarker data: $B^{\text{raw}}_{m,t,f}$.

Then for region-level biomarkers:

$$
\text{biomarkers}_{n,t,f} = \sum_{m=1}^M R_{mn} \cdot B^{\text{raw}}_{m,t,f}.
$$

Resulting tensor:

* `biomarkers` ($\in \mathbb{R}^{N \times T \times F}$)

### 3.4 Windowing for Forecasting

Given `cases` and region-level `biomarkers`, we construct samples defined by:

* A **reference time** $t_0$ such that:

  * Input window: $[t_0 - L + 1, \dots, t_0]$
  * Forecast window: $[t_0 + 1, \dots, t_0 + H]$

For each region $i$:

* Case history: $y_{i, t_0-L+1:t_0}$
* Biomarker history: $b_{i, t_0-L+1:t_0, 1:F}$
* Mobility history: $M_{t_0-L+1:t_0}$, used by MobilityGNN.

Preprocessing steps:

1. Apply any required transformations (e.g., $\log(1 + x)$ on counts).
2. Normalize features across time and/or regions.
3. Build sliding windows for training / validation / test by temporal splitting.

---

## 4. Architecture Overview

The pipeline is designed to be **inductive** with respect to the underlying region graph and mobility graph. Rather than fixing a global ordering over $N$ regions and learning a position-specific model, we operate in a **node-centric** fashion:

* The forecasting head always predicts for **one target region at a time** (or a small set), with parameters **shared across all regions**.
* Spatial and mobility context enters only via **learned embeddings and neighborhood aggregation** (GraphSAGE-style), not via fixed positional indices for nodes.
* For the MobilityGNN, we only ever load the **local neighborhood of the target region**, specifically the incoming mobility edges and their origin node features at each time step.

This inductive design allows the model to generalize to new regions or modified graphs at inference time, as long as we can provide:

* Static features and adjacency for RegionEmbedder (to compute $z_i$ for new regions), and
* Local mobility neighborhoods for MobilityGNN (incoming edges and origin node features).

The components are:

The pipeline has three main components:

1. **RegionEmbedder**

   * A GraphSAGE-style GNN over a static region graph.
   * Produces a static embedding $z_i$ for each region.

2. **MobilityGNN**

   * A per-time-step GNN over a dynamic mobility graph.
   * Uses origin–destination flows to aggregate case information from origin regions onto destination regions.
   * Produces a time-varying mobility embedding $m_{i,t}$ per region and time.

3. **ForecasterHead**

   * A Transformer-based forecasting model using temporal self-attention.
   * Consumes sequences of per-time-step features $x_{i,\tau}^{\text{forecaster}}$ and outputs forecasts $\hat{y}_{i,t+1:t+H}$.

Temporal modeling is entirely contained in the ForecasterHead. MobilityGNN is applied **independently** for each time step to produce cross-sectional mobility-enhanced features.

---

## 5. Component Design

### 5.1 RegionEmbedder

**Goal:** Learn static region embeddings $z_i$ that capture geometric and attribute-based structure of regions.

**Inputs:**

* Node features: $X \in \mathbb{R}^{N \times D_{\text{static}}}$
* Static adjacency/graph: edges connecting regions (e.g., neighbour relationships)

**Model:** GraphSAGE with $K$ layers.

For region $i$:

* Initial representation: $h_i^{(0)} = X_i$

* At layer $k$:

  $$
  h_i^{(k)} = \sigma \Big( W^{(k)} \cdot \text{AGG}\big( h_i^{(k-1)}, \{ h_j^{(k-1)} : j \in \mathcal{N}(i) \} \big) \Big)
  $$

* Final embedding: $z_i = h_i^{(K)} \in \mathbb{R}^{D_{\text{reg}}}$

**Training:**

* Objective: could be contrastive (e.g., neighbor prediction) or supervised if labels exist.
* Optimizer: Adam.
* LR scheduler: ReduceLROnPlateau or StepLR.
* Early stopping on validation metric.

**Outputs:**

* Embedding matrix `region_embeddings` ($\in \mathbb{R}^{N \times D_{\text{reg}}}$) persisted to disk.
* Optionally, RegionEmbedder weights for fine-tuning.

---

### 5.2 MobilityGNN (Per-Time-Step GNN with Mobility-Weighted Aggregation)

**Inductive, neighborhood-based operation:**

MobilityGNN is implemented in a **GraphSAGE-style inductive fashion**. For a given target region $i$ and time $t$, we only materialize the **ego-network around $i$** consisting of:

* The target node $i$
* Its immediate in-neighbors $j$ with non-zero incoming mobility flows $M_{t, j i} > 0$
* (Optionally) multi-hop neighbors if we extend to k-hop neighborhoods

The GNN then aggregates origin signals from this local neighborhood to compute the target’s mobility embedding $m_{i,t}$. The model parameters are shared across all regions and do not depend on the total number of regions $N$.

In practice, this is implemented via edge lists (e.g., `edge_index_t`, `edge_weight_t`) restricted to the incoming edges of the target node(s), rather than dense $N \times N$ matrices for the entire graph.

**Goal:** For each time step $t$, compute a mobility-enhanced feature vector $m_{i,t}$ for each region $i$ by aggregating case signals from origin regions according to mobility flows.

#### Inputs per Time Step

At time $t$:

* Case counts: $y_{i,t}$ for all regions $i$
* Optional: current biomarkers $b_{i,t,1:F}$
* Origin–destination mobility matrix: $M_t \in \mathbb{R}^{N \times N}$ where $M_{t, j i}$ is flow from origin $j$ to destination $i$

We construct node features for MobilityGNN:

$$
x_{i,t}^{\text{mob}} = \text{concat}\big( y_{i,t},\; b_{i,t,1:F},\; z_i^{\text{static?}} \big)
$$

(The inclusion of biomarkers and region embeddings into `x^{mob}` is configurable; minimally, it must include $y_{i,t}$.)

#### Mobility Graph Construction

Define an adjacency-like matrix $A_t \in \mathbb{R}^{N \times N}$ where:

* $A_t[i, j]$ measures influence from origin $j$ to destination $i$.
* We can define:

  $$
  A_t[i, j] = \text{normalize}\big(M_{t, j i}\big)
  $$

Typical normalization:

* Incoming-flow normalization per destination $i$:

  $$
  A_t[i, j] = \frac{M_{t, j i}}{\sum_k M_{t, k i} + \varepsilon}
  $$

* Optional log transform: $M'_{t, j i} = \log(1 + M_{t, j i})$ before normalization.

#### Message Passing: Mobility-Weighted Aggregation

For each time step $t$, we run one or more GNN layers:

* Initial node state: $h_{i,t}^{(0)} = x_{i,t}^{\text{mob}}$

* Single GCN/GraphSAGE-style layer:

  $$
  h_{i,t}^{(1)} = \sigma \Big( W_{\text{self}} h_{i,t}^{(0)} + \sum_{j=1}^N A_t[i, j] \, W_{\text{neigh}} h_{j,t}^{(0)} \Big)
  $$

This implements:

> Aggregate origin regions' current case-related features into the destination region using **incoming mobility weights**.

* We can stack multiple layers and use residual connections if desired.

* Define the per-time-step mobility embedding as:

  $$
  m_{i,t} = h_{i,t}^{(L_{\text{gnn}})} \in \mathbb{R}^{D_{\text{mob}}}
  $$

where $L_{\text{gnn}}$ is the number of GNN layers.

#### Temporal Aspect

* MobilityGNN is **not recurrent** across time.
* It is applied independently to each time slice $t$ with its own $x_{\cdot,t}^{\text{mob}}$ and $A_t$.
* Temporal structure is captured later by the Transformer ForecasterHead, which sees the sequence $\{m_{i,t_0-L+1}, \dots, m_{i,t_0}\}$.

---

### 5.3 ForecasterHead (Transformer with Temporal Attention)

**Goal:** Use self-attention over time to forecast future incidence for each region from sequences of per-time-step features.

#### Inputs per Sample

For each region $i$ and reference time $t_0$, we consider the sequence of length $L$:

* Case history: $y_{i, t_0-L+1:t_0}$
* Biomarker history: $b_{i, t_0-L+1:t_0, 1:F}$
* Mobility embeddings: $m_{i, t_0-L+1:t_0}$
* Static region embedding: $z_i$

At each time step $\tau \in [t_0-L+1, t_0]$ we build a feature vector:

$$
\text{local}_{i,\tau} = \text{concat}\big( y_{i,\tau},\; b_{i,\tau,1:F} \big)
$$

$$
x_{i,\tau}^{\text{forecaster}} = \text{concat}\big( \text{local}_{i,\tau},\; m_{i,\tau},\; z_i \big)
$$

Sequence for region $i$:

$$
X_i^{\text{forecaster}} = \big[ x_{i,t_0-L+1}^{\text{forecaster}},\; \dots,\; x_{i,t_0}^{\text{forecaster}} \big] \in \mathbb{R}^{L \times D_{\text{in}}}
$$

#### Transformer Architecture

* Positional encodings over time steps $1..L$
* $L_{\text{tr}}$ Transformer encoder layers with:

  * Multi-head self-attention across the time dimension
  * Feed-forward networks
  * Residual connections and layer normalization

Two options for the prediction head:

1. **Sequence-to-vector head**:

   * Pool final hidden states (e.g., take last time step or attention pooling) to get a summary vector, then map to $H$ outputs.

2. **Sequence-to-sequence head**:

   * Use a decoder or projection that outputs a sequence of length $H$ representing $\hat{y}_{i, t_0+1:t_0+H}$.

For simplicity, we can start with sequence-to-vector (e.g., use final time step embedding) and a linear head to produce $H$ predictions.

#### Loss

* Continuous regression loss on transformed cases (e.g., log-transformed):

  * Mean Squared Error (MSE) or Mean Absolute Error (MAE)
* Optionally, count-based losses (e.g., Poisson NLL) if modeling counts explicitly.

---

## 6. Model Variants

We support four primary variants to ablate the contributions of regions and mobility.

Let `use_region_embeddings` and `use_mobility_embeddings` be boolean flags.

1. **Base forecaster (temporal only)**

   * Inputs: epidemic series only (cases ± biomarkers)
   * `use_region_embeddings = False`
   * `use_mobility_embeddings = False`
   * Region embedding $z_i$ and mobility embedding $m_{i,\tau}$ are replaced with zeros.

2. **Forecaster + regions**

   * Inputs: epidemic series + static region embeddings $z_i$
   * `use_region_embeddings = True`
   * `use_mobility_embeddings = False`
   * Mobility embeddings zeroed.

3. **Forecaster + mobility (GNN)**

   * Inputs: epidemic series + mobility embeddings $m_{i,\tau}$
   * `use_region_embeddings = False`
   * `use_mobility_embeddings = True`
   * Region embeddings zeroed.

4. **Forecaster + regions + mobility (full)**

   * Inputs: epidemic series + region embeddings + mobility embeddings.
   * `use_region_embeddings = True`
   * `use_mobility_embeddings = True`

Zeroing rather than removing inputs keeps the architecture shape identical across variants.

---

## 7. Training and Evaluation

### 7.1 Data Splits

* Use **temporal splits** to mimic operational forecasting:

  * Train on earliest part of timeline.
  * Validate on a middle segment.
  * Test on the latest segment(s).
* Optionally use rolling-origin evaluation.

### 7.2 Training Loop (High-Level)

1. Pretrain RegionEmbedder (if used) to obtain static region embeddings.
2. For each model variant:

   * Initialize MobilityGNN and ForecasterHead.
   * Train jointly using forecasting loss.
   * Evaluate on validation/test sets.

### 7.3 Metrics

* RMSE, MAE over predicted vs observed incidence.
* Relative error metrics for scale-free comparison.
* Peak-related metrics (timing and magnitude of incidence peaks) as needed.
* Stratified metrics by region type or incidence level.

---

### 8 End-to-End Forward Pass for One Batch (Node-Centric, Inductive)

Assume we sample a batch of **target regions** and reference times. For each batch element $b$, we have:

* `target_region_id[b]`: the index/id of the target region $i$
* `cases_hist[b]`: $(L,)$ case history for the target region
* `biomarkers_hist[b]`: $(L, F)$ biomarker history for the target region
* For each time index in the window, a **local mobility ego-graph** around the target:

  * `node_features_t[b]`: $(N_{\text{sub}_t}, D_{\text{in}})$ local node features (cases and optional covariates)
  * `edge_index_t[b]`: $(2, E_{\text{sub}_t})$ origin/destination indices within the local subgraph
  * `edge_weight_t[b]`: $(E_{\text{sub}_t},)$ normalized mobility weights
  * an index `target_local_idx[b]` pointing to the target node within `node_features_t[b]`

## Summary

In this form, the entire pipeline is **inductive**:

* RegionEmbedder can embed new regions given their static features and neighbors.
* MobilityGNN only depends on local ego-graphs around the target node at each time step.
* The ForecasterHead always operates on sequences of features for **one region at a time**, with no dependence on the global number of regions or their ordering.

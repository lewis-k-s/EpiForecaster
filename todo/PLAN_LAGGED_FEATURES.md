# Plan: Lagged Feature Engineering for Mobility Processor

## 1. Goal
Capture "Seeding Effects" (incubation lag) without the complexity of a Spatiotemporal GNN. By explicitly feeding the GNN with "Mobility-Weighted Lagged Cases", the static GNN can learn that infection arriving from a neighbor $j$ is proportional to $j$'s prevalence $\tau$ days ago.

**Mathematical Definition:**
$$Risk_{i,t}^{(\tau)} = \sum_{j} M_{ji}(t) \cdot Cases_j(t-\tau)$$

Where:
*   $M_{ji}(t)$ is the normalized mobility flow from $j$ to $i$ at time $t$.
*   $Cases_j(t-\tau)$ is the normalized case count in region $j$ at time $t-\tau$.
*   $\tau \in \{1, 3, 7, 14\}$ are the lag windows.

## 2. Implementation Strategy

We will implement this using a **"Pre-compute & Modulate"** pattern to support both Dense and Factorized mobility efficiently.

### A. The `MobilityPreprocessor` Extension
We extend `MobilityPreprocessor` with a static method `compute_imported_risk`.
*   **Input:** `cases` (T, N, 1), `mobility_matrix` (T, N, N) or (N, N), `lags` (List[int]).
*   **Logic:**
    1.  Shift the `cases` tensor for each lag (padding with 0).
    2.  Perform Batch Matrix Multiplication: `Mobility @ ShiftedCases`.
    3.  Output: `(T, N, len(lags))`.
*   **Factorized Support:** If mobility is factorized ($M_{base}, \kappa_t$), we compute risk using $M_{base}$. The dataset loader will later apply the scalar modulation $(1 - \kappa_t)$.

### B. Integration in `EpiDataset`
*   **Initialization:**
    *   Load `precomputed_cases`.
    *   Compute `lagged_risk` using `MobilityPreprocessor.compute_imported_risk`.
    *   Store in RAM (it's small: $T \times N \times 4$ floats).
*   **`__getitem__`:**
    *   Slice `lagged_risk` for the current window.
    *   If synthetic/factorized: Apply $(1 - \kappa_t)$ modulation on-the-fly.
    *   Concatenate to `case_node` features.
*   **Config:**
    *   Update `EpiForecasterConfig` to include `mobility_lags: list[int] = [1, 7, 14]`.
    *   Update `input_dim` calculation in model to account for extra channels.

## 3. Step-by-Step Code Changes

### Step 1: Update `data/mobility_preprocessor.py`
Add `compute_imported_risk` method.
```python
def compute_imported_risk(
    self, 
    cases: np.ndarray, 
    mobility: np.ndarray, 
    lags: list[int]
) -> np.ndarray:
    # ... logic ...
```

### Step 2: Update `models/configs.py`
Add `mobility_lags` field to `DataConfig`.

### Step 3: Update `data/epi_dataset.py`
*   In `__init__`: Call `compute_imported_risk`.
*   In `__getitem__`: Slice and concat features.
*   Update `temporal_node_dim` property to reflect new channels.

### Step 4: Update `models/epiforecaster.py`
Ensure `temporal_node_dim` logic accounts for the new configuration.

## 4. Why this works (Physics)
This creates a "short-circuit" for the model to learn diffusion. Instead of needing to deduce that "Node A predicts Node A (autoregression) AND Node A predicts Node B (diffusion)", the GNN is explicitly fed "The virus pressure from neighbors 7 days ago". This makes the causal link of transmission immediate and easy to learn for a static GNN.

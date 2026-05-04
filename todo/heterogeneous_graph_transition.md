# TODO: Transition to Heterogeneous Graph Architecture

## Objective
Transition the EpiForecaster model from a homogeneous municipality graph (relying on spatial interpolation) to a **heterogeneous graph** that natively supports multiple node types (Municipalities, WWTP Catchments, Mobility Zones). This will allow for more accurate data fusion by supervising wastewater signals at their native catchment resolution while modeling epidemic spread at the municipality resolution.

## Architectural Impact & Tasks

### 1. Data Preprocessing Layer
- **Module**: `data/preprocess/region_graph_preprocessor.py`
- **Tasks**:
    - Update to maintain separate node sets for different entities.
    - Define and extract cross-type edge relationships (e.g., `municipality_belongs_to_wwtp`, `mobility_zone_intersects_municipality`).
    - Evolve the Zarr schema to store multi-type adjacencies and typed edge attributes.

### 2. Data Loading & Batching
- **Modules**: `data/epi_dataset.py`, `data/epi_batch.py`
- **Tasks**:
    - Move away from the 'dense' graph optimization (`x_dense`, `adj_dense`) which assumes homogeneous dimensions.
    - Implement a sparse `HeteroData` (PyTorch Geometric) pipeline.
    - Update `EpiDataset.__getitem__` to stop pre-interpolating biomarkers onto municipalities; instead, provide them on their native WWTP nodes.

### 3. GNN Architecture
- **Module**: `models/mobility_gnn.py`
- **Tasks**:
    - Refactor `MobilityDenseEncoder` (which uses `DenseGCNConv`) to a sparse heterogeneous architecture.
    - Utilize `torch_geometric.nn.HeteroConv` to handle message passing between different node types.

### 4. Forecaster Backbone & Heads
- **Modules**: `models/transformer_backbone.py`, `models/observation_heads.py`
- **Tasks**:
    - Update `TransformerBackbone` to handle embeddings from multiple node types or a multi-type readout.
    - Refactor `WastewaterObservationHead` to compute its loss by linking latent infected population fractions in municipalities to WWTP-level observations using the native graph edges (contribution weights).

## Benefits
- **No Interpolation Error**: Eliminates the artifacts introduced by population-weighted area-overlap interpolation.
- **Improved Interpretability**: Directly models the physical relationship between catchment areas and the communities they serve.
- **Scalability**: Heterogeneous graphs better support adding future data sources (e.g., satellite imagery, retail flows) that exist on their own spatial grids.

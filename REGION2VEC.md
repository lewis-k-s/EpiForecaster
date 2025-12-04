# REGION2VEC Design

## Objective
Region2Vec learns geospatial node embeddings that jointly encode intrinsic region descriptors, contiguity structure, and observed interaction flows. The learned embeddings should place regions with strong flows and short-path connectivity close together, while forcing distant or weakly connected regions apart. Embeddings then feed classical clustering to recover spatial communities.

## Data & Graph Preprocessing
Let the dataset describe $N$ regions. Each preprocessing run must emit:

- **Feature matrix** $X \in \mathbb{R}^{N \times F}$. For each region we compute metric-area, perimeter, population, density, longitude, and latitude (rotating them into meters in EPSG:3035 before measuring and reprojecting centroids to WGS84). These six scalars provide morphology + demographic signals. The matrix is stored as a float32 Zarr dataset `features` with dims `(region, feature)`.
- **Spatial adjacency** $A \in \{0,1\}^{N \times N}$. We build queen or rook contiguity (user-chosen) via libpysal, enforce symmetry, and later add self-loops. The adjacency is stored as edge_index `[2, E]` (dataset `edge_index`) plus metadata capturing contiguity type.
- **Flow matrix** $S \in \mathbb{R}_{\ge 0}^{N \times N}$. When mobility or OD data exist we now read the user-configured `mobility_zarr_path`, filter it to the requested `[start_date, end_date]` window (defaulting to the final 30 days in the dataset: 10 April 2021–09 May 2021), and take the mean OD flows over that window before writing dataset `flows`. When mobility data are unavailable we fall back to binary flows implied by adjacency so the ratio loss still has supervision.
- **Hop distances** $H \in \mathbb{N}^{N \times N}$. During training we compute pairwise shortest-path hops (Breadth First Search over $A$) and cache them once per dataset. Hop distances drive the spatial prior and hop-margin term.
- **Region identifiers** `region_ids`, ensuring downstream artifacts can be rejoined to GIS tables.

The `RegionGraphPreprocessor` already performs these steps: it loads GeoJSON, joins population CSV, assembles the six-feature vector, produces contiguity edges, averages mobility flows over the configured temporal window (falling back to adjacency-implied flows if mobility data are missing), and writes the Zarr group at `outputs/region_graph/region_graph.zarr`. This satisfies the data spec.

## Encoder (Region2Vec Backbone)
We adopt a two-layer GCN over the normalized adjacency $\tilde{A}=A+I$ with $\tilde{D}_{ii}=\sum_j \tilde{A}_{ij}$:
$$
Z^{(1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X W^{(0)}\right), \qquad
Z^{(2)} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} Z^{(1)} W^{(1)}.
$$
The resulting embeddings $Z\in\mathbb{R}^{N\times D}$ should be optionally normalized. In practice we use the configurable `InductiveNodeEncoder`, which generalizes this backbone (supports residuals, dropout, aggregation choices) but remains faithful to the Region2Vec intent of message passing over contiguity.

## Pair Construction Strategy
Pairs are sampled each epoch to avoid $O(N^2)$ scaling:

- **Positive set** $\mathcal{P} = \{(i,j) : S_{ij} > \tau\}$, where $\tau$ is a flow threshold (default 1.0). If flows are missing we replace $\mathcal{P}$ with undirected edges from $A$.
- **Negative set** $\mathcal{N} = \{(i,j) : S_{ij} \le \tau\}$. When only adjacency exists we leave $\mathcal{N}$ empty and rely on hop-based negatives.
- **Hop-constrained set** $\mathcal{H} = \{(i,j) : H_{ij} > h_{\text{min}}, H_{ij} \le h_{\text{max}}\}$, using configurable thresholds (default $h_{\text{min}}=2$, $h_{\text{max}}=5$).

For each set we subsample without replacement up to a user-defined budget per epoch (default 4096/4096/2048). We cache numpy arrays of candidate indices so sampling is cheap.

## Region2Vec Loss Formulation
Let $d_{ij} = \lVert z_i - z_j \rVert_2$. Define log-flow weights $w_{ij} = \log(\max(S_{ij}, \epsilon))$.

1. **Flow ratio term** encourages strongly interacting regions to contract relative to negatives:
$$
L_{\text{ratio}} = \frac{\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}} w_{ij} d_{ij}}
{\frac{1}{|\mathcal{N}|}\sum_{(u,v)\in\mathcal{N}} d_{uv} + \epsilon}.
$$
2. **Hop term** pushes distant nodes apart, scaled by hop uncertainty:
$$
L_{\text{hop}} = \frac{1}{|\mathcal{H}|}\sum_{(i,j)\in\mathcal{H}} \frac{d_{ij}}{\log(H_{ij}+1)}.
$$
3. **Primary spatial/flow loss** (optional) applies contrastive community objectives on full batches. When a flow matrix exists we use the Community Oriented Loss combining contrastive flow signals, spatial margins, and Moran-style autocorrelation weighting. Otherwise we fall back to a Spatial Only Loss over adjacency and hop neighborhoods.

Total loss per epoch is
$$
L = L_{\text{primary}} + \lambda_{r} L_{\text{ratio}} + \lambda_{h} L_{\text{hop}},
$$
with configurable weights $\lambda_{r}=1.0$ and $\lambda_{h}=0.3$ by default.

## Training Loop
Each epoch performs:
1. Forward pass through the encoder on the full graph (mini-batching is unnecessary at current scales but could be added with neighbor sampling if $N$ grows).
2. Evaluate the primary spatial/community loss using all embeddings.
3. Sample pair batches (positives, negatives, hops) and accumulate $L_{\text{ratio}}$ and $L_{\text{hop}}$.
4. Combine weighted losses, backpropagate, optionally clip gradients (default $\lVert g \rVert \le 1.0$), and apply Adam with $\eta=10^{-3}$, weight decay $5 \times 10^{-4}$.
5. Track the best checkpoint by validation loss (here training loss) and log metrics every `log_every` epochs.

## Agglomerative Clustering Stage
After training we freeze the encoder, compute $Z$, and optionally cluster regions via Ward-linkage Agglomerative Clustering with contiguity-constrained connectivity derived from $A$. The number of clusters defaults to 14 but is capped at $N$. Cluster assignments are saved alongside embeddings for downstream analyses.

## Alignment With Existing Implementation
- `data/region_graph_preprocessor.py` already carries out the required GeoJSON loading, population join, six-feature construction, contiguity edge extraction, OD-flow averaging over the configured window (with adjacency fallback), and Zarr serialization that this design references.
- `training/region_embedder_trainer.py` instantiates the configurable encoder, sampling budgets, ratio/hop losses, optional community loss, and agglomerative clustering. The `PairSampler`, loss assembly, hop-based scaling, and artifact exports conform to the math described above, so the current trainer matches the Region2Vec spec without further changes.

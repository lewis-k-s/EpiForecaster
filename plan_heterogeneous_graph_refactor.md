EpiForecaster: native-support heterogeneous GNN refactor

Status: Agreed architectural direction; implementation handover. Sections 10–12 (added 21 September 2026, after full repository verification) ground the plan in the actual codebase, freeze cross-slice interfaces, and break the work into implementation slices; where they conflict with earlier assumptions, the verified repository facts in section 10 prevail and the deviation is called out explicitly.
Date: 21 September 2026.
Scope: Spatial representation and forecast-query refactor, with the necessary data, routing, training and evaluation changes. No major transformer or SIRHD redesign.

1. End goal and rationale

Build one jointly trained forecaster that preserves native observation supports: municipal cases, ABS hospital admissions and comarca deaths, informed by mobility on native MITMA regions. The completed source-native design also accommodates EDAR wastewater observations; stage that migration separately to isolate the clinical and mobility changes.

A forecast sample represents one source population at one forecast origin, not a municipality with four geographically harmonized labels. A forward call can process multiple source types and heads together. The heterogeneous GNN supplies source-specific context; the existing temporal backbone and shared SIRHD decoder operate on each queried source sequence.

This replaces the requirement that all information pass through municipalities. Municipality, MITMA and ABS boundaries cross one another: none is a consistently finer parent partition. Overlap determines which nodes exchange information, not which source is the target.

The resulting model is a native-support forecaster with local mechanistic decoders—not a globally reconciled multiscale epidemic simulator. Preserving observation support, improving forecasts and stabilizing latent states are separate objectives.

Evidence motivating the intervention

The supplied static diagnostics justify prioritizing support correction over another loss sweep. [D1]

Case projection control: mean mobility WIS gain changes from +0.00972 against native municipal labels to −0.00891 after ABS projection and −0.03990 after comarca projection. Beneficial configurations fall from 4/4 to 1/4 and 0/4. Forecasts, dates and evaluation instances remain fixed; only geographic labels change.

Native ABS diagnostic: transformed-rate point MAE improves with mobility in all four configurations, with mean relative improvement 4.30%. This covers 85 observed ABS sources, 339 source-date observations and four weekly target dates, under 99% coverage and a single-contributing-fold requirement.

These are complementary diagnostics, not estimates of retraining gains. Municipal WIS and native point MAE are different metrics; their magnitudes are not comparable. Normal CRPS remains slightly adverse in raw-count MAE, and Normal NLL improves only 7/16 fold-seed pairs despite its positive aggregate effect. No credible native death score was available under the fold-safety requirements. No retraining or confirmation-period evaluation was used.

The geographic diagnostics report 768 municipalities, 454 MITMA regions, 379 ABS geometries and 43 comarca geometries. Population-weighted median clipped areas are approximately 32.01, 10.30, 8.43 and 484.22 km² respectively. These support a cross-cutting design, not a municipality→ABS hierarchy. They neither establish MITMA–ABS boundary alignment nor reconcile population totals. [G1]

2. Scope boundaries

Change the spatial encoder, native-source input assembly, source indexing, target sampler, batch collation, head routing and native-support scoring. Replace the current shared municipal eligibility rule with head/source-specific eligibility. The current handoff reports only 161 eligible municipal targets despite a 768-node context graph. [H1]

Retain the temporal backbone’s architecture, history/horizon conventions and common feature-width interface. Retain the full SIRHD equations, solver and established latent-training configuration initially. Input adapters, type/source embeddings and observation routing must change, but do not introduce cross-target transformer attention, independent per-type backbones or new compartment systems.

Do not bundle this work with a new loss sweep, a simulator rewrite, hard cross-geography reconciliation or a claimed cure for latent collapse. Keep SIRHD-versus-direct-decoder experiments available as later ablations, not prerequisites for this refactor.

3. Native data and graph contract

Node roles

Node type

Own native information

Real-data forecast role

SIRHD role

Municipality

Cases and compatible municipal covariates

Cases

Full rollout when queried

ABS

Hospital admission history

Admissions

Full rollout when queried

Comarca

Death history

Deaths

Full rollout when queried

MITMA

Directed OD flows and mobility summaries

Context only

No rollout required

EDAR, staged migration

Plant-level biomarker observations

Wastewater

Resolve catchment population and decoder semantics before enabling

Construct registries from actual source IDs and versions; do not hard-code geometric counts as supervised-source counts. The handoff distinguishes 378 raw ABS codes from 379 geometries, and 43 raw comarca codes from 42 mapped codes. Unmapped and unobserved sources need explicit statuses, not fabricated labels. [H1]

Keep native observations and masks at their sources. Never repopulate ABS/comarca inputs with municipal projections of the same observations. Nodes without a supervised outcome may remain valid context nodes.

Relations and weights

Use dynamic, directed MITMA→MITMA mobility edges. Connect MITMA directly and bidirectionally to municipality, ABS and comarca nodes through population-overlap relations. Retain typed clinical overlap relations such as municipality↔ABS and municipality↔comarca. Additional direct clinical relations are configurable, not a requirement for a dense all-types graph.

Derive each required geographic overlap directly from polygons and the population grid. In particular, do not obtain MITMA–ABS weights by composing crosswalks through municipalities: that would reintroduce an intermediate mixing assumption.

Store raw overlap mass, the fraction of each endpoint’s population represented, coverage, and provenance. Reverse relations share geometric overlap but need their own normalization and message function. Keep overlap-grid populations distinct from the audited population denominators used for clinical rates and SIRHD states.

Feature pooling is not count allocation. Learned aggregation may combine embeddings; it must not silently redefine observed totals. Any separate municipal-count→source operator must use municipality-normalized allocation weights, not a transpose or inverse of source-normalized source→municipality weights.

Preserve OD direction and explicitly define inbound/outbound message semantics. Audit native self-flows and inter-MITMA trips that the municipal projection placed on its diagonal. Do not automatically inherit municipal self-loop removal or sparsification thresholds; record the retained flow mass under the native policy. These are construction checks, not additional tuning campaigns.

4. Forecast-query and batch contract

Replace:

(municipality_id, forecast_origin) → four series on that municipality

with:

ForecastQuery:
    source_type
    source_id
    forecast_origin
    forecast_horizon / requested reporting intervals
    requested_heads

Resolve support version, population, allowed context and compatible heads through the dataset/configuration. Validate the request before execution.

Typical real-data samples request municipal cases, ABS admissions or comarca deaths. A single batch can contain all three. An ABS query is not selected implicitly from a municipal query; do not enumerate municipality–ABS–comarca combinations or duplicate source targets by overlap degree.

Trajectory identity is source-conditioned, not head-conditioned. For the same source, origin and permitted context, compute one latent rollout and apply the requested compatible heads. Merge repeated requests into their union of heads/intervals before computation. Deduplication also includes dataset/realization and context identity, so different synthetic runs cannot be merged. Synthetic examples with several co-supported outcomes therefore supervise one shared trajectory.

Return predictions grouped by head with source IDs, origins, reporting intervals, units, masks and population/support versions. Do not expose an ambiguous fixed four-channel municipal output as the new native API. Unsupported head/source combinations should fail validation or be explicitly marked as inferred, unvalidated outputs—not silently scored as native observations.

5. Spatial encoding and unchanged temporal interface

Use type-specific encoders for each source’s actual history, static covariates, observation masks and observation age. Project them into compatible graph feature widths without manufacturing missing outcome channels.

A bounded initial spatial block is:

Native clinical encodings
    → typed overlap messages into MITMA context nodes
    → directed message passing over native MITMA OD
    → direct overlap messages back to clinical target nodes
    + native-history residual and clinical-overlap context
    → gather queried source-node histories
    → common input adapter: [B, L, D]
    → shared temporal transformer
    → source-conditioned initial state and rates
    → shared SIRHD rollout
    → requested native observation head(s)

Use a shallow, relation-specific block with residual connections and controlled relation fusion. An ABS representation must retain its own admission history rather than becoming only an average of municipal features. Broad comarca context must not automatically overwhelm municipal histories.

Spatial information is aggregated before the transformer, into a representation that stays attached to each source node. The transformer sees one temporal sequence per queried source: node types occupy batch rows, not extra space-time tokens. The sample contract changes; the temporal attention architecture does not.

Share graph computation among queries with identical origins and permitted context. Group incompatible origins or fold masks separately. During training, reuse differentiable representations within the forward computation, not detached embeddings across optimizer updates. At inference, any embedding cache must include checkpoint, source-data and context-mask identities.

6. SIRHD, observation heads and synthetic supervision

Retain full SIRHD rollouts for all three clinical query types. Share the solver and predictor parameters, while allowing source/type context to condition initial states and effective rates. requested_heads controls observation routing, not the underlying epidemic trajectory. MITMA context nodes do not need a mechanistic decoder merely because they belong to the graph.

The local populations overlap across node types. Do not sum their states or populations across layers, or interpret membership messages as physical population transfers. Local conservation does not imply cross-support consistency. Shared parameters and graph context retain multitask learning, but real cases, admissions and deaths now usually supervise different local trajectories.

Native observation heads must preserve measurement semantics. ABS admissions observe admission inflow over the reported interval, not occupancy H or its net change. Death observations use the appropriate death increment/flow, not cumulative D as an interval label. Retain the configured weekly hospitalization support and seven-day reporting lag, applied once. [D1, H1]

Parameterize the predictive distribution at the source’s native reporting interval. Spatial aggregation of municipal predictive distributions is no longer required for the primary path. Where weekly outputs are derived from daily latents, apply the reporting operator before constructing the weekly predictive distribution. Summing marginal quantiles or scales is not a valid substitute for a specified joint distribution.

Latent stability remains a separate synthetic-training programme. The working strategy is mechanistic simulation with direct SIRHD latent supervision; this refactor makes no assumption that native clinical supervision will resolve collapse. Preserve existing latent scaling and training decisions unless changed in a separately tracked experiment.

Synthetic query labels must identify their true support. Several outcomes and latent states for one source should share one rollout. Distinguish simulator-resolved trajectories from trajectories allocated across a crosswalk; municipal-only simulations cannot supply independently resolved intra-municipal ABS labels through deterministic redistribution. Document simulator-state→SIRHD state/flow mappings: EpiSim’s documented engines use extended SEIR formulations rather than an automatically identical SIRHD state space. [E1]

7. Training, provenance and leakage controls

Store observation interval, publication/availability date, mask, transformation and source population separately. Preserve genuine zeros versus missing observations. Daily encoder grids may contain explicitly marked imputed context, but interpolation does not create additional supervised observations. Clinical targets currently use log1p(count / population × 100,000); preserve physical units before temporal/count aggregation and apply transformations afterward. [H1]

Use only information available at the forecast origin, including lagged clinical reports and mobility. Fit preprocessing statistics on training data. Test availability masking through every cross-type message path, not only the queried node’s own features.

Source polygons and OD observations may extend beyond the 768-municipality footprint. Keep full observation support distinct from clipped diagnostic geometry and partial graph coverage. Do not combine full-source counts/flows with a silently clipped denominator, or renormalize incomplete coverage to suggest completeness. A direct source forecast may use partial covariate coverage, but must expose it. The old 99% threshold belongs to the bottom-up rescoring diagnostic, not automatically to native-target eligibility.

Version native observations, geometries, populations, overlap tables, OD processing and source registries together. Record exclusions and numerical reconciliation checks. The newly recovered native ABS/comarca CSVs were outside the immutable campaign input manifest; create a new versioned manifest including them rather than rewriting the old provenance claim. [D1]

Sample by head/source eligibility. Use mixed-head batches or head-specific microbatches accumulated before an optimizer step. Compute masked mean loss within each head, then combine with explicit task weights. Count each source/origin/head/reporting-interval target once per logical example, irrespective of its overlap degree. Log observation counts and sampling frequencies so task weighting is auditable.

Redefine spatial splits at native source support. Distinguish temporal forecasting with allowed pre-origin local histories from strict spatial generalization, where local history may be withheld. Prevent a held-out source observation from reappearing as a projected copy or aggregate elsewhere. Do not mix checkpoints to manufacture source forecasts. Overlapping populations across different outcomes are not automatically duplicate labels; document the intended generalization claim and audit actual observation reuse.

8. Implementation sequence and bounded experiments

Stage 1 — Freeze evidence and data contracts. Archive the supplied diagnostics and existing campaign identities. Create native registries, observation tables, reporting/availability operators, direct overlap edges and source-aware split manifests. Reproduce the reconciliation checks. Record geometric versus observed-source coverage. Keep the confirmation period unavailable to development.

Stage 2 — Introduce source queries and a fixed-pooling baseline. Implement the sampler, mixed-batch collation, adapters, source-conditioned SIRHD dispatch, native heads and output schema. First use fixed overlap pooling with the same native histories that the learned graph will receive. This establishes correct support and batching before attributing gains to learned relations.

Stage 3 — Enable the heterogeneous spatial encoder. Replace fixed pooling with shallow typed messages and native MITMA mobility. Keep clinical queries, backbone, SIRHD settings, observation models and loss fixed. Verify direct MITMA–ABS context paths, native-history residuals, gradients and computation reuse.

Stage 4 — Run matched calibration comparisons. Compare fixed pooling against learned heterogeneous messages; compare projected municipal mobility context against native MITMA context; and include mobility-on/off pairs. Remove or neutralize all declared mobility-derived information in the off condition, not merely OD edge weights. Separate these comparisons sequentially rather than launching an unrestricted factorial sweep. Hold native targets, histories, splits, seeds, loss and synthetic-training regime matched; report capacity, optimizer exposure and compute differences.

Report native WIS, point error, interval coverage and WIS components by head and horizon. Keep equal-source transformed-rate metrics distinct from raw-count/population-sensitive metrics. Report source and fold-seed variability; do not treat repeated origins or overlapping supports as independent replications. Use common source/date subsets for direct comparisons and report expanded coverage separately. Existing municipal WIS is a historical diagnostic, not a directly comparable primary native score.

Stage 5 — Complete EDAR migration separately. Add plant queries using validated catchment/contribution semantics, assay masks, normalization and reporting operators. Do not equate wastewater concentration with additive clinical counts or assume an administrative population is the catchment population. Until then, keep the legacy wastewater branch and its support explicitly identified, with the same treatment in paired clinical experiments; do not label the system fully source-native.

The SIRHD/direct-decoder ablation and changes to synthetic latent training remain separate from these spatial comparisons. Access the untouched confirmation period only after the evaluation protocol and selected design are frozen.

9. Acceptance criteria and remaining implementation choices

The refactor is complete for the clinical prototype when:

Queries and batching are correct: one call handles municipal, ABS and comarca queries; deterministic single-query and mixed-batch outputs agree within tolerance; head selection does not change the same source’s latent rollout.

Spatial support is preserved: native inputs and labels retain their source IDs; ABS can receive distinct native MITMA context without a municipal bottleneck; overlap coverage and relation directions are tested.

Observation semantics are correct: hospitalization intervals/lag, rate denominators, transforms, zero/missing masks and source exclusions pass unit tests. Unsupported head/source requests fail explicitly.

Training remains connected and auditable: native losses reach the intended head, SIRHD, transformer and graph relations; source deduplication and per-head weighting work; synthetic multi-head/latent supervision still uses a shared source rollout.

Evaluation is valid and reproducible: source-aware splits and as-of masking prevent leakage, native metrics and manifests are reproducible, inference cost is reported, and the confirmation period remains unused during selection.

Choose and freeze the remaining implementation details before the corresponding experiment: source-population reconciliation and partial-coverage policy; direct-overlap normalization and sparsification; clinical relation set and message depth; OD self-flow treatment; fixed loss and task sampling weights; and EDAR catchment/decoder semantics. These choices must not reopen the agreed source-query contract or force a common geographic hierarchy.

Success means correct native-support forecasting with a reusable shared temporal/mechanistic backbone. Improved native forecast quality is the experimental objective, not a guaranteed acceptance outcome; latent-collapse resolution and globally coherent populations are not claims of this refactor.

Evidence and source notes

[D1] Supplied static-diagnostics summary, this conversation. Case projection control, fold-safe ABS rescoring, WIS decomposition, operator/provenance checks and calibration-only use of 2020-12-31–2021-01-27. These results are taken from the supplied report, not rerun for this plan.

[G1] Supplied MITMA/ABS resolution and boundary-alignment diagnostics, this conversation. Based on the 768-municipality membership in ../EpiForecaster/data/processed/real_with_id.zarr; clipped source geometries and GEOSTAT-grid population weights. Establishes MITMA as native mobility support.

[H1] Pasted text(1).txt — “Spatial-resolution handoff — real canonical dataset”. Native clinical supports, current transformations, masks, eligibility and coverage. Its 768×768 mobility description concerns the processed municipal tensor; [G1] provides the upstream native-geography clarification.

[E1] EPISIM.md. EpiSim interface and extended-SEIR engine documentation; motivates an explicit synthetic state/flow mapping rather than assuming SIRHD equivalence.

Architectural requirements above consolidate the discussion. Sections 10–12 resolve repository module names, raw native file locations, and v1 source-population and split policies against the implementation repository — including the upstream `dasymetric_mob` source project (§10.2b) for native MITMA OD, zone geometry, the GEOSTAT grid, and official comarca boundaries. Remaining policy upgrades and excluded work are listed in §11.7 and §12.3.

---

## 10. Repository grounding (verified 2026-09-21)

Every claim in this section was verified against the worktree `heterogeneous-graph-refactor` (commit `9e0ea7e`) or, where noted, the sibling study checkout `/Volumes/HUBSSD/code/meprecisa/EpiForecaster`. Line numbers are from the current files and will drift; symbol names are the stable anchors.

### 10.1 What a sample is today, end to end

| Stage | Module | Contract |
|---|---|---|
| Sample identity | `EpiDataset.__getitem__` (`data/epi_dataset.py`) | One sample = one `(target_node, window_start)` pair on the 768-municipality grid; `EpiDatasetItem` carries 3-channel `[value, mask, age]` histories for hosp/deaths/cases, biomarker block, four `(H,)` targets + masks. |
| Eligibility | `get_valid_window_starts_dict` + `DataConfig._resolve_min_observed_map` (`data/epi_dataset.py`, `models/configs.py`) | Per-target cadence from `TARGET_OBSERVATION_CADENCE_DAYS` (cases 1, hospitalizations 7, deaths 1, wastewater 7); count thresholds `ceil(ceil(window/cadence)·fraction)`; node-level `valid_targets` density mask from preprocessing. |
| Graph | `optimized_collate_graphs` (`data/epi_batch.py`) + `inject_gpu_mobility` / `inject_context_features` (`utils/training_utils.py`, called from `utils/device.py`) | PyG `Batch` is a placement carrier only; dense `x_dense [B·T, N_ctx, F]` and `adj_dense [B·T, N_ctx, N_ctx]` are injected batch-side on GPU. |
| Spatial encoder | `MobilityDenseEncoder` (`models/mobility_gnn.py`) | Dense GCN/GAT over per-timestep municipal adjacency; target-node gather returns `[B, T, 64]` appended to the backbone input. |
| Temporal encoder | `TransformerBackbone.forward(x_seq [B, T, in_dim])` (`models/transformer_backbone.py`) | Unchanged by this refactor; width arithmetic in `EpiForecaster._get_backbone_input_dim` / `_get_temporal_node_dim` (`models/epiforecaster.py`). |
| Mechanism | `SIRRollForward` (`models/sir_rollforward.py`) | Fraction space, `population=ones(B)`; returns `S/I/H/R/D_trajectory [B, H+1]`, `hospitalization_flow`/`death_flow [B, H]`. Retained as-is. |
| Observation heads | `ClinicalObservationHead` ×3 + `WastewaterObservationHead` (`models/observation_heads.py`) | loc/scale in `log1p(per-100k)`; ww/hosp/cases emit `[B, H+1]` (t=0 nowcast), deaths `[B, H]`; deaths head has anchoring hardcoded off. |
| Loss | `JointInferenceLoss` (`evaluation/losses.py`) | Four hard-coded head tables; `_build_supervision_weights` gates per `(sample, head)` over `[B, H]`; `compose_total_loss` stacks exactly a 4-vector; GradNorm `task_names = ('ww','hosp','cases','deaths')` (`training/gradnorm.py`). |
| Eval | `evaluate_loader` (`evaluation/eval_loop.py`), `granular_export.py` (schema v3), `aggregate_export.py` | Four copy-pasted per-head blocks; granular row per observed `(split, target, node_id, window_start, horizon)` point; selection score `mean_target_crps` with NLL/MAE fallbacks (`training/epiforecaster_trainer.py:191-209`). |

### 10.2 Verified input inventory and what is actually native

| Path (`data/files/`) | Support / cadence | Native? | Role in this refactor |
|---|---|---|---|
| `Registre_de_casos_de_COVID-19_a_Catalunya_per_municipi_i_sexe.csv` | Municipality, daily (`TipusCasData`, `MunicipiCodi`, `NumCasos`) | Yes | Municipal cases head — already native; reuse existing processor output. |
| `COVID-19__Persones_hospitalitzades.csv` | **ABS, weekly epi-week** (`data_inici`/`data_final`, `codi_abs`, summed `casos`, native `poblacio` per ABS) | Yes | ABS admissions head. Currently *unused* by the pipeline. |
| `Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv` | **Comarca, daily** (`Data defunció`, `Codi Comarca`, `Nombre defuncions`) | Yes | Comarca deaths head. Currently *unused* by the pipeline. |
| `hospitalizations_municipality.csv`, `deaths_municipality.csv` | Municipality (weekly hosp / daily deaths) | **No — pre-projected** | Current pipeline inputs ("Already weighted by polygon overlap", `deaths_processor.py` docstring). Superseded for native configs; retained for paired projected-context experiments (§8 Stage 4). |
| `mobility.zarr` | 451 days (2020-02-14→2021-05-09) × **963×963**, float64 | **Projection** | Product of the dasymetric project (§10.2b): MITMA→municipality OD projection. Zone IDs are 5-char municipal INE codes (945/947 match `municipality_id_reference.csv`); diagonal self-flow present (≈32% of mass on 2020-02-14). |
| `geo/fl_municipios_catalonia.geojson` | ~965 municipal polygons, EPSG:3042, `id` = INE 5-digit string | Yes | Municipal geometry (study-period vintage); comarca dissolve cross-check (§10.3(2)). |
| `geo/cartografia/ABS.shp` (+ AGA/RS/SS) | ICS health cartography; ABS = basic health areas | Yes | ABS geometry (the 379-geometry support in [G1]). |
| `mpiscatalunya.csv` | Municipality→comarca mapping (4-line header) | Yes | Comarca membership (partition); processor exists (`municipality_mapping_processor.py`) but is not wired into the pipeline. |
| `fl_population_por_municipis.csv` | Municipal population (`id`, `d.population`) | Yes | Audited per-100k denominator for municipal cases and SIRHD; also the vaccination denominator. |
| `edar_muni_edges.nc`, `wastewater_biomarkers_icra.csv`, `EDAR.geojson` | EDAR plants, contribution matrix | Yes | Legacy wastewater branch — untouched, explicitly identified (Stage 5 is separate). |
| `data/processed/real_with_id.zarr` | run_id=`real`, T=435 (2020-03-01→2021-05-09), N=768 municipal `region_id` strings | Canonical training store | Municipal cases histories/context come from here; native products are written to a **new** store, never in place. |

### 10.2b Upstream source project: `dasymetric_mob` (verified read-only)

The project that produced `mobility.zarr` lives at `/Volumes/HUBSSD/code/meprecisa/dasymetric_mob` (own AGENTS.md; treat as **read-only source**, never modified from this programme). Its `output/mobility_results.zarr` is dimension-identical (451×963×963, same dates/coords) to EpiForecaster's `data/files/mobility.zarr`. It contains everything previously marked "blocked on input acquisition":

| Path (under `dasymetric_mob/data/`) | What it is | Verified facts | Use here |
|---|---|---|---|
| `mitma_v1/*.daily_mobility.v1.recurrent.csv.gz` | **Native MITMA daily OD**, long format | 451 files, exactly 2020-02-14→2021-05-09; columns `date, origin, destination, activity_origin, activity_destination, distance, residency, age, study_*, sex, income, trips, trips_km`; zone IDs like `01001_AM` (district-level malla codes) | Primary mobility graph (slice `MOB-2`): filter to Catalonia, `sum(trips)` over all strata (same aggregation the upstream pipeline used, `mobility_processor.py` pivot `aggfunc="sum"`) |
| `mitma_v1/zonificacion-distritos/distritos_mitma.shp` | **Native MITMA zone geometry** | 2850 Spain-wide districts; 442 carry Catalan-province INE prefixes (08/17/25/43) | MITMA context-node geometry; zone-count reconciliation (442 prefix-filter vs [G1]'s 454 vs transform attr 458) resolved in `MOB-2` audit |
| `mitma_v2/mitma_v2/zonificacion/{relacion_ine_zonificacionMitma.csv, poblacion.csv}` | INE↔MITMA zone mapping; per-district population | pipe-separated; section↔district↔municipality↔MITMA district/municipality/GAU | MITMA node registry and populations |
| `GEOSTAT_grid_POP_1K_2011_V2_0_1.csv` + `GRID_GEOM/*.shp` | 1-km population grid + ETRS89-LAEA grid cell polygons | grid IDs like `1kmN2689E4337` with population; **2011 vintage** (same grid [G1] and the existing transform used) | Grid-weighted overlaps (primary policy, slice `OVL-1`) |
| `IGCG_divisiones_administrativas/divisions-administratives-v2r1-*-20250730.shp` | **Official Catalan boundaries v2r1** (2025-07-30): comarques, municipis, provincies, vegueries at 1:5k–1:1M | [G1] used `comarques-50000` | Primary comarca geometry (slice `GEO-1`); municipal-dissolve becomes cross-check |
| `output/matrices/{T_ext,T_int,W_area}[_v2].{npz,zarr}` | Precomputed dasymetric source→municipality weight matrices | — | Provenance reference only; not consumed |
| `mitma_v2/estudios_basicos/por-distritos/{viajes,personas,pernoctaciones}` | v2 basic-studies product, 2022-01-01 onward (544 files) | Outside the study window (2020-03-01→2021-05-09) | **Unused** |
| `nommon_mobility_indicator/` | Nommon mobility-indicator product (28GB) | Different product, coarser semantics | **Unused** |

`compare_target_geography_resolution.py` in that project is the [G1] diagnostics script itself (reads `../EpiForecaster/data/processed/real_with_id.zarr`, `cartografia/ABS.shp`, `distritos_mitma.shp`, `comarques-50000`, GEOSTAT-weighted stats) — the upstream provenance of §1's geographic diagnostics is now fully identified.

### 10.3 Corrections and precisions to sections 1–9

1. **`mobility.zarr` is not native MITMA OD — but the native OD exists upstream.** Its attrs ("Transformed mobility matrix from source to target geometries", legacy shape `[963, 458]`) record the dasymetric MITMA→municipality projection, which happened in the sibling project `dasymetric_mob` (§10.2b). The plan's "native MITMA regions" are therefore realizable now: `mitma_v1` daily recurrent OD (451 days, exactly matching the study window) + `distritos_mitma.shp` zone geometry + GEOSTAT grid. v1 of this refactor makes native MITMA-zone OD the primary mobility graph (slice `MOB-2`), with the 963-zone municipal projection retained as a comparability artifact (slice `MOB-1`). Zone counts need an explicit audit: 442 districts carry Catalan INE prefixes, [G1] reports 454 in-footprint zones, and the stored transform attr says 458 — the `MOB-2` build fixes the zone set (prefix filter ∪ spatial intersection with the study footprint) and records the reconciliation.
2. **Comarca geometry is present after all** — official IGCG v2r1 `comarques` polygons live in `dasymetric_mob/data/IGCG_divisiones_administrativas/` ([G1] used the 1:50k set). Caveat: v2r1 is dated 2025-07-30; the municipal-dissolve cross-check (from the study-period `fl_municipios_catalonia.geojson` + `mpiscatalunya.csv`) remains in `GEO-1` to surface boundary-vintage differences. ABS polygons (`ABS.shp`) do **not** partition municipalities — municipality↔ABS overlap always requires true polygon intersection.
3. **WIS does not exist in `evaluation/`.** The repo scores NLL, closed-form Normal CRPS, and interval coverage/width. All [D1] WIS numbers come from the diagnostics script `analyze_source_resolution.py` (in the study bundle, §10.4) computing WIS + dispersion/under/over components from granular interval columns. Stage-4 native WIS reporting therefore requires a small, tested WIS module in `evaluation/metrics.py` (slice `EVL-1`).
4. **ABS CSV structure** (revised after full-file verification): rows are per (epi-week, ABS, sex, age group); 311 Catalonia-total sentinel rows coincide exactly on `codi_abs=="000"` ∧ `codi_regio=="00"` ∧ "No disponible" ∧ `index_socioeconomic==-1` ∧ `poblacio==0`. **`index_socioeconomic==-1` alone is NOT a sentinel** — 6 real newer ABS carry it with genuine admissions (870 in-window casos); the sentinel filter is the structural triple only. **`poblacio` is per-stratum and case-only** (only case-bearing strata are reported; median 2 rows per ABS-week), so the CSV cannot supply an ABS population denominator: `obs_admissions` stores `stratum_pop_sum` for QC only, and the audited ABS denominator comes from overlap-implied municipal population at `native_graph` assembly (OVL-1/GRF-1). Epi weeks are Monday–Sunday. Native-vs-projected weekly sums differ by a real vintage drift (measured: median 4.0, max 13, total 281 counts over 54 common weeks — the projected file predates a CSV revision), so the OBS-1 integration cross-check asserts median < 20 counts and records the distribution; [D1]'s 1e-5 figure was an operator re-projection check, not a file comparison. In-window observed ABS = 377 (the [D1] "85" figure is a coverage/fold-filtered diagnostic subset, not a registry rule).
5. **Eligibility mechanics** are exactly the machinery this refactor generalizes: `TARGET_OBSERVATION_CADENCE_DAYS` already encodes weekly hospitalization support, so per-head native cadence thresholds fall out of the existing resolver; the [H1] 161-eligible-target figure is the `valid_targets` cases-density rule (≥0.5), which belongs to the municipal path and is **not** inherited as a native-source rule (§7).
6. **Processed-store writer drift**: current preprocessing code writes NetCDF (`real_with_id.nc`); `data/processed/real_with_id.zarr` is the canonical DVC artifact consumed by training. New native stores are written with `xarray.to_zarr` (zarr v3) under `data/processed/native_support/` as new versioned artifacts — never patched in place (immutable-artifact rule).
7. **Footprint vs support**: the 963 OD zones vs 768 study municipalities is an intersection artifact (cases∩mobility∩population>0). Mobility context may legitimately use the full 963-zone support with an explicit coverage mask (§7: full observation support distinct from clipped footprint).
8. **Model seam is narrower than it looks**: ww/hosp/cases/deaths heads already map 1:1 onto municipality/ABS/comarca(+legacy EDAR) source types. The refactor re-targets *which source node* each head supervises; the head modules, SIRHD solver, backbone, loss combiner (4-vector), GradNorm task set, and checkpoint-selection chain are reused. Weekly ABS supervision adds one new component (interval reporting operator, slice `ROP-1`) rather than a new head family.
9. **Death-head anchoring** is already disabled in code (`anchor_mode="disabled"`, `delta_forecasting=False` hardcoded for the deaths head in `models/epiforecaster.py`), consistent with §6's death-increment semantics.
10. **Confirmation period**: [D1] used 2020-12-31–2021-01-27 for calibration only. Native evaluation code must refuse to score that window unless an explicit guard flag is set (slices `MAN-1`, `EVL-1`).

### 10.4 Baseline studies for Stage 4 (user-provided)

Most recent matched baselines: `/Volumes/HUBSSD/code/meprecisa/EpiForecaster/outputs/loss_research/loss-type-full-coverage-v2` — a separate-worktree loss-family campaign with **8 conditions** `{normal, student_t} × {crps, nll} × {mobility_on, mobility_off}` replicated over fold·seed pairs, plus `checkpoint_reuse_receipt.json` / `reuse_evidence_v1` provenance and the `source_resolution_analysis/` bundle that reproduces the [D1] numbers (verified: `case_projection_control` and `hospital_native_source` blocks match §1 exactly). For this refactor's comparisons, filter to the production-supported subset `normal:nll`, `normal:crps` × mobility on/off; `student_t` rows are excluded. The bundled `scripts/analyze_source_resolution.py` is the reference implementation for WIS components and fold-safe native rescoring.

---

## 11. Frozen cross-slice interfaces

These interfaces are fixed for the whole implementation programme. Slices implement against them; a slice that needs a change escalates it back for review rather than improvising.

### 11.1 Source types, heads, cadence (`data/query_contract.py`)

```python
SourceType = Literal["municipality", "abs", "comarca", "mitma", "edar"]
# Real-data query routing (supervised heads only; synthetic may request unions):
HEADS_FOR_SOURCE = {
    "municipality": frozenset({"cases"}),
    "abs": frozenset({"hosp"}),
    "comarca": frozenset({"deaths"}),
    "mitma": frozenset(),       # context-only, never queried
    "edar": frozenset({"ww"}),  # legacy branch only, until Stage 5
}
REPORTING_CADENCE_DAYS = {"cases": 1, "hosp": 7, "deaths": 1, "ww": 7}
```

Head names stay exactly the existing canonical set `ww | hosp | cases | deaths` used by `evaluation/losses.py`, `granular_export.py`, and `aggregate_export.py`; the weekly ABS variant is surfaced as `hosp` scored at weekly reporting intervals, not as a fifth head name. The weekly-`hosp` loss replaces the daily-`hosp` component inside the existing 4-vector slot (§11.5), so GradNorm's task tuple and `compose_total_loss` semantics are unchanged.

### 11.2 `ForecastQuery`

```python
@dataclass(frozen=True)
class ForecastQuery:
    source_type: SourceType
    source_id: str                 # native registry ID ("08121", "088", "06", ...)
    forecast_origin: int           # index into the shared daily date coord
    horizon_days: int              # default config.model.forecast_horizon
    requested_heads: frozenset[str]
    realization_id: str = "real"   # dataset/run identity — synthetic runs never merge
```

`validate_query(q, registry)` fails on: unknown `(source_type, source_id)`; empty `requested_heads`; head not in `HEADS_FOR_SOURCE[source_type]`; unobserved source status (explicit `UnsupportedQueryError`, never silent). `merge_queries(qs)` unions heads over identical `(source_type, source_id, forecast_origin, realization_id)` and refuses heterogeneous identities.

### 11.3 Offline artifact schemas (`data/processed/native_support/`)

All stores are zarr v3 written via `xarray.to_zarr`, string IDs as coords, storage dtypes per `data/dtypes.py`:

- `obs_admissions.zarr` / `obs_deaths.zarr` — dims `(date, source_id)` (admissions indexed by `week_start` with aligned `week_end` coord): `value` (raw counts, float32), `mask` (bool, true observation only — genuine zeros are value 0 with mask 1), `age` (uint8, reporting opportunities since last observation). `population` (float32, audited denominator) is stored for deaths (summed-municipal); admissions store `stratum_pop_sum` (float32, QC-only lower bound) — the audited ABS denominator is assigned at `native_graph` assembly from overlap-implied municipal population. Municipal cases need no obs store: `DS-1` reads the cases block (already log1p per-100k) and temporal covariates directly from `real_with_id.zarr`. No per-100k/log transforms at rest in the native stores — transforms are applied at consumption and recorded in metadata.
- `overlap_edges.parquet` — one row per directed overlap pair: `src_type, src_id, dst_type, dst_id, overlap_area_m2, overlap_pop_est, frac_of_src_pop, frac_of_dst_pop, coverage_src, coverage_dst, provenance` (provenance = fixed string recording geometry versions + weighting policy). Municipality↔comarca entries are exact memberships (frac 1.0) because comarcas partition municipalities.
- `od_mitma.zarr` — **primary mobility graph**: native MITMA-zone OD built from `dasymetric_mob/data/mitma_v1` (slice `MOB-2`): dims `(date, origin, target)` over the audited Catalan MITMA zone set, `mobility` = raw summed trips float32 at rest (log1p/threshold applied at consumption), zone registry (ID ↔ `distritos_mitma.shp` ↔ `poblacion.csv`), audit attrs: zone-count reconciliation, `diag_mass_fraction`, retained mass per policy, date completeness (451 days). Built store (verified): 451 days × 456 zones (442 prefix + 14 footprint-only), `dates_missing=[]`, deterministic across rebuilds. Zone populations use the frozen exact-`distrito` join only (349/456 available; 107 NaN because the v2 `poblacion.csv` zonification is finer than v1 zones and `_AM` districts key differently) — accepted for v1 because MITMA nodes are context-only; the v1↔v2 population crosswalk (`relacion_ine_zonificacionMitma.csv`) is a recorded MAN-1 QC item, not a blocker. Never fabricate municipal-sum populations for subdivided zones.
- `od_municipal.zarr` — comparability artifact built from the in-repo `mobility.zarr` (slice `MOB-1`): the 963-zone municipal-coded projection, log1p float16, same audit attrs; consumed only by the projected-context condition of Stage-4 comparisons.
- `native_graph.zarr` — node table (`node_type, source_id, population, status ∈ {observed, unobserved, unmapped}`) + per-relation edge groups with endpoint-normalized weights.
- `native_manifest_v2.json` — sha256 of every input file (including the two native clinical CSVs, closing [D1]'s provenance gap), code-version stamp, schema versions, reconciliation results, `confirmation_period: ["2020-12-31", "2021-01-27"]` marker.

### 11.4 `NativeEpiBatch` (`data/native_batch.py`)

```python
@dataclass
class NativeEpiBatch:
    queries: list[ForecastQuery]              # length B, order-defining
    hist: torch.Tensor                        # (B, L, F_native) unified history features
    hist_mask: torch.Tensor                   # (B, L) observed-validity for the query's own series
    source_type_id: torch.Tensor              # (B,) int enum index
    source_pop: torch.Tensor                  # (B,) audited denominator for per-100k transform
    origin_dow: torch.Tensor                  # (B,) weekday of forecast_origin (0=Mon)
    ctx_feat: torch.Tensor                    # (B, L, D_ctx) fixed-pooling context (Stage 2) / hetero encoder input (Stage 3)
    temporal_covariates: torch.Tensor         # (B, L, k), shared with the municipal path
    # Per-head supervised blocks, gathered by routing index (length B_h ≤ B):
    head_index: dict[str, torch.Tensor | None]      # {"cases","hosp","deaths"} row positions
    head_target: dict[str, torch.Tensor | None]     # cases/deaths: (B_·, H) daily; hosp: (B_h, W) weekly
    head_target_mask: dict[str, torch.Tensor | None]
    head_population: dict[str, torch.Tensor | None] # per-head denominator (ABS native weekly population)
```

### 11.5 Reporting-operator semantics (weekly ABS)

Given daily admission-flow trajectory `f_{1..H}` (fraction space) from the SIRHD rollout and source population `P_w` for epi-week `w`: mechanistic weekly location = `log1p( (Σ_{t∈w} f_t) · P_w · 1e5 )`; sum **before** transform (§6). Weekly intervals are complete Mon–Sun weeks fully inside `(origin, origin+H]`; partial weeks get mask 0. Anchoring/delta-forecasting for the weekly head is disabled in v1 (frozen choice, mirrors the deaths head). Scale: obs-context-driven exactly like other heads. The predictive distribution and all scores live in weekly `log1p(per-100k·week)` space; raw-count MAE is derived by inverse transform with `P_w` recorded.

### 11.6 Configuration surface

New dataclasses in `models/configs.py` (defaults on the dataclass; no runtime fallbacks): `NativeSupportConfig` (`enabled: bool = False`, `root: str = "data/processed/native_support"`, `spatial_encoder: Literal["pooled","hetero"] = "pooled"`, `od_source: Literal["mitma","municipal"] = "mitma"`, `overlap_weighting: Literal["geostat_grid","uniform_density"] = "geostat_grid"`, `od_self_flows: Literal["retain","drop"] = "retain"`, `od_min_flow_raw: float = 10.0`, `head_balanced_sampling: bool = True`, `confirmation_guard: bool = True`), plus native-builder keys in `data/preprocess/config.py` (including `mitma_source_root`, an explicit path to the read-only `dasymetric_mob` checkout, required only for `MOB-2`). Native mode disables curriculum mixing and the legacy projected municipal hosp/deaths inputs (asserted, not silently ignored). Bounded local acceptance smoke config: `configs/train_epifor_native_smoke.yaml`; substantive runs remain MN5-only.

### 11.7 Frozen v1 policy choices (upgrade paths noted)

- **Overlap weighting**: GEOSTAT 1-km grid weighting (grid cells + populations from `dasymetric_mob`, §10.2b) as the primary policy — the same weighting [G1] and the existing dasymetric transform used, which keeps our overlaps consistent with both; uniform-municipal-density areal weighting is the fallback when the grid is unavailable. Both keep overlap-grid populations distinct from audited denominators (§3). Known limitation: the grid is 2011-vintage; recorded in provenance, consistent with every existing artifact derived from it.
- **Mobility graph**: native MITMA-zone OD (`od_mitma.zarr`) is primary; the 963-zone municipal projection (`od_municipal.zarr`) is retained for the projected-context comparison condition. OD direction always origin→target; self-flow policy and thresholds are consumption-time, recorded per §11.3 audits.
- **Absent-row policy for registries**: absent `(date, source)` rows are missing (mask 0), never zero; genuine zeros only where the source reports them. QC reports absence rates; revisit only with evidence.
- **Splits**: municipal folds from the existing representative manifest; ABS/comarca fold = fold owning the majority of the source's population overlap; exactly-once per source type validated. Generalization claim documented as "held-out source, visible context history" (temporal-forecasting mode, §7).
- **Source conditioning**: source-type embedding (+ optional per-source static embedding) concatenated into the backbone input; direct initial-state/rate conditioning deferred.
- **Synthetic data**: continues on the municipal path; native-mode training is real-data-only in this programme. Synthetic-native unification (shared-rollout multi-head queries, [E1] state mapping) is a separate follow-up.

---

## 12. Deliverable slices

### 12.0 Ground rules for the implementation loop
Each slice is one goal-loop unit: read `AGENTS.md`, implement only the listed files, add only the listed tests, run `uv run ruff check .` and the slice's tests (`uv run pytest tests/test_<slice-file>.py -q`), and commit as `native: <slice-id> <imperative summary>`. Never modify `data/files/` inputs or existing processed stores in place; new artifacts go to `data/processed/native_support/` (git-ignored; DVC tracking optional later). The sibling project `/Volumes/HUBSSD/code/meprecisa/dasymetric_mob` is a **read-only source**: consume its files, hash them into the manifest, never write there. Interfaces in §11 are contracts — if a slice cannot satisfy one, stop and report rather than redesign. No GPU, no cluster submission, no formatter-wide rewrites, no edits outside the slice's file list. Sizes: S ≤ ~150 lines + tests; M ≤ ~400 lines + tests.

### 12.1 Slice dependency graph

```
Data/contract track                Model/training/eval track
---------------------              --------------------------
QC-1  (independent)                ROP-1 (independent)
REG-1 (independent)
GEO-1 (independent)
OBS-1 (independent)
OBS-2 (independent)
MOB-1 (independent)
MOB-2 (independent; needs read-only dasymetric_mob checkout)

OVL-1  <- GEO-1
GRF-1  <- REG-1, GEO-1, OVL-1, OBS-1, OBS-2, MOB-1, MOB-2
MAN-1  <- GRF-1
DS-1   <- GRF-1, QC-1
BAT-1  <- DS-1
MOD-1  <- BAT-1, ROP-1, QC-1
LOS-1  <- MOD-1
SPL-1  <- GRF-1
TRN-1  <- LOS-1, SPL-1
EVL-1  <- LOS-1
ENC-1  <- MOD-1, TRN-1
CMP-1  <- EVL-1, ENC-1
CUT-1  <- everything, after the review checkpoint
```

Independent starts (safe to parallelize): `QC-1`, `REG-1`, `GEO-1`, `OBS-1`, `OBS-2`, `MOB-1`, `MOB-2`, `ROP-1`.

### 12.2 Slice specifications

#### QC-1 — Query contract (S)
**Files**: new `data/query_contract.py`; new `tests/test_native_query_contract.py`.
**Change**: `SourceType`, `HEADS_FOR_SOURCE`, `REPORTING_CADENCE_DAYS`, `ForecastQuery`, `UnsupportedQueryError`, `validate_query(query, registry)`, `merge_queries(queries)` per §11.1–11.2. Registry access via a minimal `Protocol` (`source_status(type, id)`) so QC-1 has no dependency on REG-1.
**Tests**: validation failures (unknown source, empty heads, head/source mismatch, unobserved source); merge unions heads and refuses mixed identities (different `realization_id`, different origins); real-data routing table matches §11.1 exactly.
**Done when**: tests green; module imported by nothing else yet.

#### REG-1 — Native source registries (M)
**Files**: new `data/native_registry.py`; new `tests/test_native_registry.py` (+ tiny CSV fixtures under `tests/fixtures/native_support/`).
**Change**: `SourceRecord(source_type, source_id, status, population, first_date, last_date)`, `SourceRegistry`; builders: ABS registry from `COVID-19__Persones_hospitalitzades.csv` (sentinel filter `codi_abs=="000"` / `codi_regio=="00"` / "No disponible"; sum `casos` per (abs, week); per-ABS `poblacio` dedup + consistency check); comarca registry from the deaths CSV + `mpiscatalunya.csv` membership + summed municipal population; municipal registry from `fl_population_por_municipis.csv` ∩ `municipality_id_reference.csv`. Status assignment: `observed` (≥1 observation in the study window), `unobserved`, `unmapped` (geometry/code without observations or vice versa) — must surface the 378-vs-379 ABS and 43-vs-42 comarca discrepancies from §3 as explicit statuses, not errors.
**Tests**: fixture-driven parsing incl. sentinel filtering; population consistency tolerance; discrepancy statuses; reported zeros stay `observed`.
**Done when**: registries build from the real files when present (integration test `skipif` file absent) and from fixtures otherwise.

#### GEO-1 — Comarca geometry (S)
**Files**: new `data/preprocess/native_sources/comarca_geometry.py`; new `tests/test_native_comarca_geometry.py`.
**Change**: primary geometry = official IGCG v2r1 comarques polygons from `dasymetric_mob/data/IGCG_divisiones_administrativas/divisions-administratives-v2r1-comarques-50000-20250730.shp` (read-only source; copy the layer into `data/files/geo/generated/comarques_igcg_v2r1.geojson` in a metric CRS with 2-digit comarca IDs); cross-check = dissolve of the study-period `fl_municipios_catalonia.geojson` by `mpiscatalunya.csv` membership; report ID-set differences and per-comarca area discrepancies between the two vintages (boundary drift 2020 vs 2025) plus membership/area closure of the dissolve.
**Tests**: toy 4-municipality/2-comarca fixture verifies dissolve partition, ID propagation, area closure; real-file integration `skipif` asserts the code set matches the deaths CSV, flags codes without geometry as `unmapped`, and records (not fails on) vintage differences.
**Done when**: comarca polygon set + reconciliation report exist; no input geometry modified.
#### OVL-1 — Overlap engine and edge tables (M; depends GEO-1)
**Files**: new `data/preprocess/native_sources/overlap.py`; new `tests/test_native_overlap.py`.
**Change**: pairwise polygon intersection (municipality↔ABS from `ABS.shp`; municipality↔comarca; plus MITMA-zone↔municipality/ABS/comarca from `distritos_mitma.shp` — all direct, in a common metric CRS, **never** composed through municipal crosswalks, §3); weights per §11.7 — `geostat_grid` primary: allocate each zone's population via 1-km grid cells (`GEOSTAT_grid_POP_1K_2011_V2_0_1.csv` + `GRID_GEOM` polygons), overlap weight = shared grid population; `uniform_density` fallback. Port the minimal weighting routine (area + grid allocation) with attribution to `dasymetric_mob/dasymetric_transform.py` rather than importing across repos; grid files are read-only inputs with hashes in the manifest. Emit `overlap_edges.parquet` per §11.3 with both endpoint fractions and coverage; additional direct relations (e.g. ABS↔comarca) configurable but off by default.
**Tests**: toy rectangles verify areas, grid-weight allocation on a synthetic 2×2 grid, fractions ≤ 1, `frac_of_dst_pop` column-sums ≤ 1 per destination with deficit = uncovered population recorded; reciprocity (identical shared population both directions, different endpoint fractions per §3); real-file integration `skipif` asserts every observed ABS and every MITMA zone has nonzero municipal overlap.
**Done when**: parquet exists with provenance stamp (geometry versions + weighting policy + grid vintage); invariant tests green.
#### MOB-1 — Municipal-projection OD store (S)
**Files**: new `data/preprocess/native_sources/od_municipal.py`; new `tests/test_native_od_municipal.py`.
**Change**: read `data/files/mobility.zarr` (963 municipal-coded zones, 451 days); write `od_municipal.zarr` per §11.3 (log1p float16, zone registry from origin coords, reconciliation vs the municipal registry: matched/unmatched counts); audit attrs: diagonal mass fraction, retained mass under `{retain, drop} × {min_flow_raw}` policies, per-policy edge counts (§3 construction checks — no tuning); coverage mask for zones outside the cases footprint (§10.3(7)). Comparability artifact only — consumed by the projected-context Stage-4 condition.
**Tests**: synthetic 4×4×3 tensor verifies transform correctness, audit math (diag fraction, retained mass), registry reconciliation; direction preservation (origin→target never symmetrized).
**Done when**: store + audit written; no threshold baked into the stored tensor (policies recorded only).
#### MOB-2 — Native MITMA OD harmonization (M)
**Files**: new `data/preprocess/native_sources/od_mitma.py`; new `tests/test_native_od_mitma.py`.
**Change**: harmonize `dasymetric_mob/data/mitma_v1/*.daily_mobility.v1.recurrent.csv.gz` (451 daily long-format files, exactly 2020-02-14→2021-05-09) into a single `od_mitma.zarr` per §11.3 — this is the file-type-consistency step: per-date gzipped stratified CSVs → one chunked zarr tensor. Build: zone set = `distritos_mitma.shp` zones with Catalan-province INE prefixes (08/17/25/43; 442 verified) ∪ zones spatially intersecting the study footprint, with the resulting count and the 442/[G1]-454/attr-458 reconciliation recorded; per date, filter rows to the zone set, `sum(trips)` over all stratification columns (activity/distance/residency/age/sex/income — same aggregation the upstream pipeline used), pivot to dense `[Z, Z]` raw-trips float32; registry joins zone ↔ geometry ↔ `poblacion.csv` population; date-completeness check (451 days, missing days explicit); source root from `mitma_source_root` config/CLI — the upstream repo is **read-only**.
**Tests**: fixture with two synthetic daily CSVs verifies parsing, strata summation, zone filtering, pivot densification, missing-day detection; `skipif` integration test over a `--limit 5` real-file subset verifies totals and that trips sum to the raw file sums; full-build runs as a script, not in tests.
**Done when**: `uv run python -m data.preprocess.native_sources.od_mitma --source-root <dasymetric_mob>` writes the store + audit; unit tests green.

#### OBS-1 — Native ABS admissions processor (M)
**Files**: new `data/preprocess/processors/abs_admissions_processor.py`; new `tests/test_native_abs_processor.py`.
**Change**: mirror `hospitalizations_processor.py` structure; output `xr.Dataset` dims `(date=week_start, abs_id)` with `week_end` coord: `value` (weekly summed raw admissions), `mask`, `age` (in reporting opportunities, cap 14), `stratum_pop_sum` (per §10.3(4): per-stratum case-only sums — QC only, NOT a denominator; warn on >1% week-to-week deviation). Sentinel filter = structural triple only (never `index_socioeconomic==-1` alone). No smoothing, no daily interpolation — weekly intervals are native (§4).
**Tests**: weekly placement on fixture (interval → `week_start`, mask only on reported weeks); sex×age aggregation; stratum-pop summation + deviation warning; zero-vs-missing distinction; window cropping.
**Done when**: emits the §11.3 `obs_admissions` schema; `skipif` integration test cross-checks native weekly sums vs the projected municipal file with the §10.3(4) vintage-drift tolerance (median < 20 counts, report max/total) and asserts the in-window observed-ABS count in the 50–500 band (real: 377).

#### OBS-2 — Native comarca deaths processor (S)
**Files**: new `data/preprocess/processors/comarca_deaths_processor.py`; new `tests/test_native_comarca_deaths_processor.py`.
**Change**: mirror `deaths_processor.py`; parse `Data defunció` (DD/MM/YYYY), `Codi Comarca` (string, leading zeros), `Nombre defuncions`; aggregate sexes; output dims `(date, comarca_id)`: `value`, `mask` (reported rows only — absent rows missing per §11.7), `age` (days, cap 14), `population` (summed municipal).
**Tests**: fixture parsing, sex aggregation, absent-date = missing, string-ID preservation, window cropping.
**Done when**: emits the §11.3 `obs_deaths` schema.

#### GRF-1 — Native graph store assembly (M; depends REG-1, OVL-1, OBS-1, OBS-2, MOB-1, MOB-2)
**Files**: new `data/preprocess/native_sources/build_native_graph.py`; new `tests/test_native_graph_store.py`.
**Change**: assemble `native_graph.zarr` (§11.3): node table with statuses and populations, now including `mitma` context nodes from the MOB-2 zone registry; edge groups per relation (`muni_abs`, `muni_comarca`, `mitma_muni`/`mitma_abs`/`mitma_comarca` overlaps from OVL-1, `od_mitma` referencing the MOB-2 store) with endpoint-normalized weights; validation asserting node-type counts, edge fraction bounds, and that every `observed` clinical node and every MITMA zone has ≥1 overlap edge.
**Tests**: assembled from small synthetic inputs — counts, normalization invariants, missing-node statuses; integration `skipif` over real artifacts.
**Done when**: store builds deterministically (fixed coord ordering) with a validation report.

#### MAN-1 — Manifest v2 and QC (S; depends GRF-1)
**Files**: new `scripts/analysis/native_support_qc.py`; new `data/preprocess/native_sources/manifest.py`; new `tests/test_native_manifest.py`.
**Change**: `native_manifest_v2.json` per §11.3 (sha256 all inputs incl. the two native clinical CSVs — recording that they sit outside `catalonia_preprocessing_inputs_v1` — plus the consumed read-only `dasymetric_mob` inputs: `distritos_mitma.shp`, IGCG comarques layer, GEOSTAT grid CSV + grid geometry, and the 451 `mitma_v1` daily files with an aggregate manifest when full hashing is impractical); QC report: ABS native population vs overlap-implied municipal population, comarca population closure, MITMA zone-count reconciliation (442/454/458), OD mass retention per policy on both OD stores, observed-source coverage vs geometry counts (§7); confirmation-period marker.
**Tests**: manifest determinism; hash changes when an input changes; QC tolerance checks on synthetic numbers.
**Done when**: `uv run python scripts/analysis/native_support_qc.py --root data/processed/native_support` emits manifest + report from real artifacts.

#### ROP-1 — Weekly interval reporting operator (M)
**Files**: new `models/reporting_operators.py`; new `tests/test_native_reporting_operators.py`.
**Change**: `weekly_interval_sum(daily [B,H] or [B,H+1], origin_dow [B]) -> (sums [B,W], week_mask [B,W])` mapping complete Mon–Sun weeks inside `(origin, origin+H]` (vectorized, no Python loop over B); `weekly_log1p_per100k_location(sums, population) -> [B,W]` per §11.5; partial weeks → mask 0; nowcast-inclusive inputs drop t=0 first.
**Tests**: hand-computed 14-day cases (origins at each weekday; partial weeks; resulting W values); sum-before-transform property pinned by an explicit example (operator of sum ≠ transform of per-day sums); nowcast exclusion; population scaling.
**Done when**: property tests green; no dependency on any other slice.

#### DS-1 — Native query dataset (M; depends GRF-1, QC-1)
**Files**: new `data/native_dataset.py`; new `tests/test_native_dataset.py`.
**Change**: `NativeSupportDataset(native_root, real_with_id_path, config)`: per-head `(source, origin)` indexes with head/source-specific eligibility — reuse `DataConfig._resolve_min_observed_map` semantics at each head's native cadence (cases daily municipal, hosp weekly ABS counting complete weeks, deaths daily comarca); histories per source type: municipal cases from `real_with_id.zarr` cases block (+ shared temporal covariates); ABS weekly series on a daily grid with interval placement, mask and week-age channels; comarca daily 3-channel; fixed-pooling `ctx_feat` via `overlap_edges` weights over municipal daily cases (+ OD summary row, config-gated); `__getitem__` returns items matching §11.4 minus collation; exposes `head_counts` for the §7 sampling audit.
**Tests**: synthetic native-store fixture (4 muni, 2 ABS, 2 comarca, ~40 days): per-head index eligibility (weekly counting for ABS), history shapes/channels, context pooling weights sum to 1 per query with coverage deficit exposed, query dedup (identical query appears once).
**Done when**: dataset yields deterministic items; reads only the native root + `real_with_id.zarr`.

#### BAT-1 — Native batch and collation (S; depends DS-1)
**Files**: new `data/native_batch.py`; new `tests/test_native_batch.py`.
**Change**: `NativeEpiBatch` per §11.4 + `collate_native_batch(items)`: unified `[B, L, F_native]` (source-type one-hot block + per-type channel blocks, zero-padded cross-type slots), per-head routing indices, `origin_dow`, per-head population; `to(device)` / `pin_memory()` mirroring `EpiBatch`.
**Tests**: mixed three-type batch: shapes; routing disjointness and coverage (each query in exactly its head's index); padding zeros carry mask 0; weekday computation from the shared date coord.
**Done when**: collate is pure (no I/O) and tested.

#### MOD-1 — Native forecaster adapter (M; depends BAT-1, ROP-1, QC-1)
**Files**: new `models/native_forecaster.py`; new `tests/test_native_forecaster.py`.
**Change**: `NativeInputAdapter` (source-type embedding + linear projection to the backbone width; width from config, bypassing base width arithmetic); `NativeEpiForecaster` **composing** the existing `EpiForecaster`: instantiate base with `ModelVariant(cases=False, biomarkers=False, mobility=False, regions=…)` so the base contributes backbone + SIRHD + four heads unchanged; native forward = adapter → `base.backbone` → `base.sir_rollforward` → cases/deaths heads as today → weekly `hosp` location via ROP-1 on `hospitalization_flow` (+ residual/scale through the existing `hosp_head` obs-context path; anchoring disabled per §11.5). Output dict = the existing key set plus `dist_hosp_weekly_loc/scale [B_h, W]` and `pred_hosp_weekly`, with query metadata echoed for routing. One rollout per query row; all heads computed and routed downstream (mathematically identical to per-query head selection).
**Tests** (CPU, seeded): single-query vs mixed-batch outputs identical within 1e-6 for the same query; changing `requested_heads` leaves `S/I/H/R/D` trajectories bit-identical; weekly keys align with ROP-1 hand examples; gradients reach adapter, backbone, and every head path.
**Done when**: the §9 "Queries and batching are correct" criterion is demonstrated by these tests.

#### LOS-1 — Routed loss integration (M; depends MOD-1)
**Files**: modify `evaluation/losses.py` (additive subclass; municipal-path behavior untouched); new `tests/test_native_losses.py`.
**Change**: `NativeJointInferenceLoss(JointInferenceLoss)`: accepts routed head blocks; gathers per-head `[B_h, ·]` targets/masks via routing indices; weekly `hosp` replaces the daily `hosp` slot in the 4-vector stack (§11.1) using the same masked NLL/CRPS scoring in weekly space; eligibility per head at native cadence; `compose_total_loss` semantics (equal active split or GradNorm weights against `obs_weight_sum`) unchanged; per-head observation counts logged (§7 audit).
**Tests**: synthetic routed batch — per-head losses match manually computed masked scores; inactive head (no eligible rows) zero-weighted and weights renormalized; gradients flow to each head's parameters; count-once property (a target counted once per logical example irrespective of overlap degree — construct a duplicated query and assert a single contribution).
**Done when**: the §9 "Training remains connected and auditable" criterion holds at loss level.

#### SPL-1 — Source-aware splits (M; depends GRF-1)
**Files**: new `data/native_crossval.py`; new `tests/test_native_splits.py`.
**Change**: `native_source_folds.json`: load the existing representative municipal manifest (`configs/crossval/representative_10fold_mobility_refined.json`), derive ABS/comarca fold membership by majority population overlap (§11.7); validate exactly-once per source type, fold count, ID hashing (mirror `data/crossval_manifest.py` patterns under a new version key); leakage audit: no supervised source in two folds; the generalization claim recorded in the manifest.
**Tests**: derivation on synthetic overlaps (majority; deterministic tie-break by source ID); exactly-once violations raise; real-manifest integration `skipif`.
**Done when**: fold manifest generated with audit report; §9 source-aware-split checks pass.

#### TRN-1 — Trainer wiring and smoke run (M; depends LOS-1, SPL-1)
**Files**: modify `training/dataloader_factory.py`, `training/epiforecaster_trainer.py` (native branch behind `NativeSupportConfig.enabled`; municipal path intact), `models/configs.py` (§11.6); new `configs/train_epifor_native_smoke.yaml`; new `tests/test_native_trainer_wiring.py`.
**Change**: native dataloader over the union per-head index with `head_balanced_sampling` (deterministic round-robin head interleave under seed); trainer asserts curriculum disabled and legacy projected hosp/deaths inputs absent in native mode; smoke config bounded (tiny steps, CPU-runnable per `train_epifor_smoke.yaml` conventions).
**Tests**: sampler head composition balanced and order deterministic under seed; config parsing/defaults; a 2-step forward+backward+loss smoke on the synthetic fixture (finite losses; no NaN-halt).
**Done when**: `uv run train epiforecaster --config configs/train_epifor_native_smoke.yaml` completes locally bounded.

#### EVL-1 — Native evaluation, WIS, and exports (M; depends LOS-1)
**Files**: modify `evaluation/metrics.py` (WIS accumulators), `evaluation/granular_export.py` (schema v4: add `source_type`, `source_id`; keep `region_id` for municipal rows), `evaluation/aggregate_export.py` (three native heads incl. weekly hosp), `evaluation/eval_loop.py` (native branch: routing-aware accumulators; confirmation-period guard honoring `confirmation_guard`); new `tests/test_native_eval.py`.
**Change**: WIS + dispersion/under/over from interval columns, porting the math of the §10.4 reference script (not the file); per-head, per-horizon (weekly hosp: per-week) WIS/CRPS/point metrics keyed by `(source_type, source_id)`; equal-source transformed-rate metrics kept separate from raw-count MAE (inverse transform with recorded denominators); checkpoint-selection chain unchanged, now over three native heads.
**Tests**: WIS accumulator vs hand-computed cases including the component-decomposition identity; zero-width intervals; mask-0 rows excluded; schema-v4 columns and metadata sidecar version bump; confirmation-period guard raises unless explicitly disabled.
**Done when**: a smoke eval over the synthetic fixture writes granular v4 + aggregate CSVs + WIS summary; §9 evaluation subchecks pass.

#### ENC-1 — Heterogeneous typed-overlap encoder (M; depends MOD-1, TRN-1)
**Files**: new `models/hetero_encoder.py`; new `tests/test_native_hetero_encoder.py`; modify `models/configs.py` (`spatial_encoder: "hetero"` activates).
**Change**: `TypedOverlapMessage`: per relation (muni↔ABS, muni↔comarca, clinical↔MITMA context) learned gated aggregation over overlap-normalized neighbor features with a residual — an ABS node keeps its own admission history (identity residual; adds context, never averages into it, §5); directed OD message passing reuses `MobilityDenseEncoder` over native MITMA-zone adjacency from `od_mitma.zarr` (`od_source: "mitma"`), with the `od_municipal` projection available behind the same flag for the paired condition; bounded depth per §5's block (typed hop into MITMA context → one OD layer → typed hop back to clinical targets) plus native-history residual and clinical-overlap context; replaces pooled `ctx_feat` when enabled; identical-origin queries share the graph representation within a forward (§5 reuse rule).
**Tests**: shape/route checks; zero-init gates reproduce the pooled baseline exactly (pinned equivalence test); gradient flow through every relation block; parameter count bounded by config.
**Done when**: `spatial_encoder: pooled|hetero` A/B both produce valid training steps; equivalence-at-zero-init test green.

#### CMP-1 — Stage-4 matched-comparison configs and analysis (M; depends EVL-1, ENC-1)
**Files**: new `configs/experiment/native_stage4/` (matched pairs: pooled vs hetero × mobility on/off × {nll, crps} × OD context {native MITMA zones, projected municipal}); targets/splits/seeds/loss frozen per §8; new `scripts/analysis/native_stage4_comparison.py`; new `tests/test_native_stage4_comparison.py`.
**Change**: comparison script reads native granular v4 exports + the baseline study (§10.4, filtered to `normal:nll`/`normal:crps` × mobility on/off), computes per-head/per-horizon WIS components, point error and coverage with fold-seed pairing on common source/date subsets, and emits explicit non-comparability caveats vs historical municipal WIS; the mobility-off condition neutralizes all declared mobility-derived inputs (OD context, imported risk) per §8; the §8 "projected municipal mobility context vs native MITMA context" comparison is realized as the OD-context condition pair (`od_source: municipal` vs `mitma`); outputs a machine-readable table + summary JSON under `outputs/reports/native_stage4/`.
**Tests**: pairing/grouping and subset restriction on synthetic CSVs; caveat/metadata fields present.
**Done when**: the script runs end-to-end over smoke-scale artifacts; production execution is MN5-only (configs documented accordingly).

#### CUT-1 — Cutover and documentation (S; after the review checkpoint)
**Files**: modify native-facing `configs/`, `EPIFORECASTER.md`, `AGENTS.md` cross-references, `data/preprocess/README.md`; retire the projected municipal hosp/deaths CSVs from native training paths (files remain for paired projected-context experiments per §8); remove dead mid-programme adapters.
**Change**: single clean cutover for the clinical prototype: native support is the default real-data interface; the legacy municipal four-channel path remains available only behind explicit config for paired experiments and the untouched wastewater branch (Stage 5 stays open).
**Done when**: docs state the new contract; no orphaned code paths; `uv run ruff check .` and `uv run pytest tests/ -q` green.

### 12.3 Explicitly out of this programme

EDAR/plant migration (Stage 5), synthetic-native unification ([E1] state mapping), direct initial-state/rate conditioning, `mitma_v2` estudios-basicos ingestion (2022+, outside the study window), the Nommon mobility-indicator product, the SIRHD-vs-direct-decoder ablation, and any latent-stability programme change (§2, §6). GEOSTAT 2011-vintage grid accuracy and IGCG 2025 boundary-vintage drift are recorded limitations with cross-checks (§11.7, `GEO-1`), not blockers. None of the excluded items may be silently pulled into a slice.
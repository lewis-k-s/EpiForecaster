# Data audit for the loss research loop

Status: source audit complete; replacement preprocessing is not yet executed.
The scientific run gate remains closed until its new data identity and all-head
support are checked. Existing `real_with_id` results are diagnostics only.

## Findings and controls

- `real_with_id.zarr` contains 435 daily dates, 2020-03-01 to 2021-05-09,
  and 768 municipalities. These dates have been studied before. There is no
  untouched prospective test period and no archived release-vintage stream.
- The stored configuration uses Kalman and Tobit Kalman preprocessing. All four
  processors fitted variance parameters on full series. A training window cutoff
  did not constrain this earlier fit. The new config limits parameter fitting
  to 2020-09-30. A separate scaler defect in `EpiDataset` is also fixed.
- The smoothers initialized from the first future positive observation before
  stepping through time. This could populate earlier missing inputs from the
  future. Initialization now occurs when that observation arrives. EDAR limit
  filling now uses preceding values only. Tests perturb future values and check
  that earlier outputs and fitted parameters do not change.
- Cases are reported by municipality. The primary event threshold can use one
  observed municipality/date value. Repeated forecast windows must not count
  as new threshold-fit observations.
- Hospitalization CSVs contain 1,201 municipality/source pairs across 944
  municipalities and only 10 distinct `(codi_regio, codi_ambit)` pairs. Allocated
  municipality totals are not independent source observations. The source CSV
  places weekly totals at week start. The research preprocessing shifts these
  totals seven days forward, so the full week has elapsed. Actual release delays
  are still unknown; this does not establish real-time availability.
- Death totals are allocated from larger areas. The available municipality
  reference includes comarca membership, but the allocated CSV does not retain
  the full allocation lineage. Treat its municipalities as dependent.
- The EDAR contribution file has 59 source plants and 200 municipality entries.
  Shared plants and mixtures make municipality wastewater series dependent.
- Use a conservative uncertainty unit: contiguous target-date blocks containing
  **all regions and heads**, with block length at least 28 days. Do not use a
  region-level independent bootstrap or treat folds as independent trials.
  The short selection/calibration periods limit independent temporal support.
  A narrow interval from resampling rows is not evidence for promotion.
- Keep hospitalization event classification secondary and unsupported until an
  unambiguous source-level event and forecast aggregation are defined. Do not
  deduplicate distinct allocated values under one source identifier. Cases
  remain the primary classified high-incidence proxy.

## New input and output identities

`loss_research_raw_inputs_v1.dvc` identifies copied raw inputs, with an input
SHA-256 manifest, under the ignored research output directory. Its DVC hash is
`c65715e80dcc5947eb665c7f037ed9ed.dir` (9 files, 232,743,467 bytes). The original
input files and the existing processed dataset are unchanged.

`configs/production_only/preprocess_mn5_loss_research.yaml` defines the new
`real_loss_fit20200930_v1` dataset. The immutable workflow writes its output to
a new run root and publishes a new DVC object. It does not replace
`real_with_id.zarr` or its pointer. All losses must use the same new dataset.

The preprocessing fit-date and hospitalization-date changes alter the baseline.
Retrain the Normal-NLL control; old checkpoints are not matched controls for
this dataset. Historical-vintage and allocated-source limits remain explicit.

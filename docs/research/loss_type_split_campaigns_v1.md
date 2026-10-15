# Separate loss-type evaluation designs

| Campaign | Fit and selection | Scored support | Processed data |
| --- | --- | --- | --- |
| Temporal | All eligible regions; fit labels 1 May–21 Aug 2020; selection labels 19 Sep–21 Oct 2020 | All eligible regions; 19 Nov 2020–10 Mar 2021, four contiguous 28-day blocks | Per-series smoothing parameters fitted no later than 21 Aug 2020 |
| Spatial | Ten persisted region folds; each held-out region is a test target in one fold | All eligible target dates after the input history, through 9 May 2021 | Fixed smoothing variances; all declared wastewater variants retained |

The temporal campaign keeps 8 Apr–9 May 2021 for confirmation. Its four scored blocks share one fit cutoff and one processed dataset. A later rolling-origin campaign with distinct fit cutoffs would need a processed dataset for each cutoff, unless parameter fitting moves into the fold runtime.

The Student-t CRPS rolling-origin extension trains a separate model for each
28-day temporal block. The four fit cutoffs are 21 Aug, 18 Sep, 16 Oct, and
13 Nov 2020. Each fit has a later 33-day selection interval, a 28-day gap,
and one 28-day calibration block. Per-series smoothing parameters use only
observations through that fit cutoff. The later three datasets are prepared
locally from the same audited raw inputs and have separate DVC identities.
Each temporal campaign has paired mobility conditions, seeds 52 and 53, and
50 epochs with early stopping disabled. The suite also runs the Student-t
CRPS spatial campaign over all ten persisted folds. The complete matrix is
16 temporal fits and 40 spatial fits.

The spatial campaign uses one processed dataset for all folds. Its fixed smoothing mode does not fit Kalman variances from any municipality or wastewater source. It also avoids ranking wastewater variants by full-record prevalence. The biomarker scaler is still fitted from training regions inside each model fold; mobility edge thresholds are fixed. The OD-free region embedding is a precomputed static descriptor of the full map. This is a transfer test for known regions with observed input history, not a cold-start test for a region absent from every source.

The spatial campaign includes March–May 2021. The temporal confirmation interval is therefore protected only within the temporal campaign; it is not unexamined across the research program. Shared wastewater plants and spatial context make region results dependent. Report regional effects descriptively and use source groups and contiguous time blocks for inferential intervals.

Canonical paired tables use `fold=-1` for the temporal campaign. That value means that no region fold exists; it is not a spatial fold index.

Production campaigns are `configs/campaigns/loss_type_temporal_v1.yaml` and `configs/campaigns/loss_type_spatial_v1.yaml`. The temporal smoke is `loss_type_temporal_smoke_v2.yaml`; the spatial smoke is `loss_type_spatial_smoke_v3.yaml`. These bounded execution checks do not support model selection. They validate the emitted `wis_median` aggregate column; production analysis uses matched `mean_target_wis` from complete granular support.

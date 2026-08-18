# Synthetic SIRHD reporting-v1 corpus

This document defines the only synthetic corpus supported for latent-supervised
EpiForecaster pretraining.

## Versioned inputs

- Raw: `data/files/raw_synthetic_observations_sirhd_reporting_v1_cb6b4e5.zarr`
  (`10d13dace7243bded69c7b04defb7bdf.dir`)
- Processed:
  `data/processed/synthetic_sirhd_reporting_v1_cb6b4e5_holt_damped.zarr`
  (`2eb591ae43fe4f505120603965427404.dir`)
- Generator source commit: `cb6b4e5ed7dcf3b1a0b23a15ac5a2377e09857c1`

## Data contract

- Clinical observations are continuous expected reported counts. They are
  calibrated in `log1p(count / population * 100000)` space, then stored as raw
  counts without rounding or integerization.
- Missingness uses explicit masks and NaNs. In this controlled corpus, clinical
  no-event zeros are missing.
- Mechanistic compartments are continuous `float32` population states.
- Wastewater is raw concentration and receives one downstream `log1p`.
- Mechanistic truth is separate from the observation layer. Exported S/I/H/R/D
  targets support synthetic pretraining, but can be an incomplete partition of
  the upstream compartment model.
- Reporting delays and spikiness are disabled. A discrete Poisson or
  negative-binomial reporting variant requires its own seed and version.

The corpus contains 20 deterministic baseline mechanistic trajectories with no
intervention changes. Only the observation and reporting layer differs from the
superseded corpus. Latent truth is a synthetic-pretraining target, never an
observed real-data feature. Report model SIRHD mass constraints separately from
the mass represented by exported targets.

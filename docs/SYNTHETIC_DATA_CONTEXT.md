# Synthetic SIRHD corpora

This document defines the synthetic corpora supported for latent-supervised
EpiForecaster pretraining.

## Supported corpus: closed-loop reactive n100

- Bundle: `epiforecaster_sirhd_v1_netcdf_d9c5bae_20261015_reactive_n100`
  (200 runs: 100 Reactive profiles and their no-policy Baseline twins;
  456 dates from 2020-02-09; 945 regions; 59 EDARs).
- Raw pointer: `data/files/epiforecaster_sirhd_v1_netcdf_d9c5bae_reactive_n100.nc.dvc`
  (`a30e21844967255fa43a15157e49ccfa`, 133,926,669,691 bytes). It is a
  Git-backed import of EpiSim.jl `cd60c25`, created with `--no-download`.
  The payload stays on GPFS; put the object into the EpiForecaster DVC store
  there by hard link, not through a workstation.
- Generator source commit: `d9c5bae998bf666cd9022cd019d3e195d36cb17e`.
- Preprocessing: `workflow/config/mn5_preprocess_sirhd_v1_reactive_n100.yaml`.
- Pretraining: `workflow/config/mn5_synth_pretrain_reactive_n100.yaml`.

### Latent state-space contract

Preprocessing reads only the generator's model-aligned SIRHD partition. The
primitive `latent_*_true` stocks are audit-only and are never SIRHD targets.

| Processed target | Raw source | Primitive stocks |
|---|---|---|
| `latent_S_true` | `sirhd_S` | S |
| `latent_I_true` | `sirhd_I` | E + A + I + PH + PD |
| `latent_H_true` | `sirhd_H` | HR + HD |
| `latent_R_true` | `sirhd_R` | R |
| `latent_D_true` | `sirhd_D` | D |
| `latent_death_flow_true` | `sirhd_death_flow` | D[k+1] - D[k], dated at date[k+1] |

The preprocessing config pins `synthetic_state_space_schema_version` and
`synthetic_state_space_contract_sha256`. The processor recomputes the SHA-256
of the embedded canonical contract JSON and stops on a mismatch. It also
requires S+I+H+R+D to equal the population, the confined stock to be zero, and
a death-flow interval for every processed date. The first corpus date has no
incoming interval, so processing starts on 2020-02-10.

Corpora without `sirhd_*` stocks, such as the reporting-v1 corpus below, are
rejected for latent supervision.

## Superseded corpus: SIRHD reporting-v1

### Data contract

- Clinical observations are continuous expected reported counts. They are
  calibrated in `log1p(count / population * 100000)` space, then stored as raw
  counts without rounding or integerization.
- Missingness uses explicit masks and NaNs. In this controlled corpus, clinical
  no-event zeros are missing.
- Mechanistic compartments are continuous `float32` population states.
- Wastewater is raw concentration and receives one downstream `log1p`.
- Mechanistic truth is separate from the observation layer. This corpus has
  no `sirhd_*` partition; its primitive S/I/H/R/D stocks omit E, A, PH and PD
  from I and cannot be used under the current latent contract.
- Reporting delays and spikiness are disabled. A discrete Poisson or
  negative-binomial reporting variant requires its own seed and version.

The corpus contains 20 deterministic baseline mechanistic trajectories with no
intervention changes. Only the observation and reporting layer differs from the
superseded corpus. Latent truth is a synthetic-pretraining target, never an
observed real-data feature. Report model SIRHD mass constraints separately from
the mass represented by exported targets.

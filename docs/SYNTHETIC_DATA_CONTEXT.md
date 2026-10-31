# Synthetic SIRHD corpora

This document defines the synthetic corpora supported for latent-supervised
EpiForecaster pretraining.

## Supported corpus: closed-loop reactive n100 (a97c812, Zarr)

This is the only synthetic corpus. Its DVC pointer is its identity.

- Bundle: `epiforecaster_sirhd_v1_zarr_a97c812_20261027_reactive_n100`
  (200 runs: 100 Reactive profiles and their no-policy Baseline twins;
  456 dates from 2020-02-09; 945 regions; 59 EDARs). The seed cap
  `max_seed_half_region_population` v1 leaves no run excluded.
- Raw pointer:
  `data/files/epiforecaster_sirhd_v1_zarr_a97c812_reactive_n100.zarr.zip.dvc`
  (`cfdd2a02c7a81a8f9e008339b693b1a5`, 46,612,607,993 bytes). It is a
  Git-backed import of EpiSim.jl `323d78a`. The payload stays on GPFS; put the
  object into the EpiForecaster DVC store there by hard link, not through a
  workstation.
- Generator source commit: `a97c812ffbf5331aae7ae0e5f844fcb55dbca205`.
- Preprocessing: `workflow/config/mn5_preprocess_sirhd_v1_a97c812_reactive_n100.yaml`.
  It writes `data/processed/synthetic_sirhd_v1_a97c812_reactive_n100_holt_damped.zarr.zip`.
  Promote the generated processed pointer after review.
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

Corpora without `sirhd_*` stocks are rejected for latent supervision.

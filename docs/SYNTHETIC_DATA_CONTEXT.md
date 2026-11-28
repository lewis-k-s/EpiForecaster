# Synthetic SIRHD corpora

This document defines the synthetic corpora supported for latent-supervised
EpiForecaster pretraining.

## Supported corpus: closed-loop reactive n100 (a97c812, Zarr)

Two corpora are supported. Each one's DVC pointer is its identity.

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

## Supported corpus: closed-loop reactive n400 (5a2262d, Zarr)

- Bundle: `epiforecaster_sirhd_v1_zarr_5a2262d_20261105_reactive_n400`
  (440 runs: 400 Reactive runs, each with a policy, and 40 no-policy Baseline
  twins; only profiles 0, 10, 20, ... have a twin; 456 dates from 2020-02-09;
  945 regions). The sampled parameter ranges are wider than for n100: R0
  log-uniform 1.0-1.9, wider severity scales, and log-uniform case
  ascertainment in reporting profile `sirhd_pretraining_reporting_v4`.
- Raw pointer:
  `data/files/epiforecaster_sirhd_v1_zarr_5a2262d_reactive_n400.zarr.zip.dvc`
  (`afda29837a0e7c77b48e86a5429f6812`, 239,065,027,536 bytes). It is a
  Git-backed import of EpiSim.jl `2d05b2d`. Hard-link the object on GPFS from the
  EpiSim.jl DVC store into both `EpiForecaster-DVC/storage` and
  `EpiForecaster-DVC/cache`: `dvc pull` does not fetch an import's object, so
  without the cache link `stage_data` fails with "DVC output was not
  materialized".
- State space `epiforecaster_sirhd_v2`, contract SHA-256
  `2e133e67d9278a9e05e2689961f17339497440530d5a88deee106323abe6ca54`.
  It has the same five stocks and destination-dated death flow as v1 and adds
  `sirhd_infection_flow` and `sirhd_hospitalization_flow`, which preprocessing
  does not read. A config pinned to v1 rejects it.
- Generator QA: all reactive gates and the factorized mobility parity pass;
  19 of 55 reporting-QA checks fail against real-data targets and are waived in
  EpiSim.jl `qa_acceptance.json`.
- Preprocessing: `workflow/config/mn5_preprocess_sirhd_v1_5a2262d_reactive_n400.yaml`.
  It writes `data/processed/synthetic_sirhd_v1_5a2262d_reactive_n400_holt_damped.zarr.zip`.
  Run `preprocess-5a2262d-reactive-n400-20261106e` (job 48637267) published the
  promoted pointer, DVC object `5b465d72394fba15addce1dd3c493754`
  (204,212,147,324 bytes, 440 runs). Run `preprocess-5a2262d-reactive-n400-20261106d`
  (job 48632495) published the same content under `fcc42650acd1c7568d51df9cc8c42789`;
  the md5 differs only through the zip member timestamps.
- Pretraining: `workflow/config/mn5_synth_pretrain_reactive_n400.yaml`, ten
  epochs on `acc_bscls` with the n100 settings. The two-epoch run
  `synth-pretrain-5a2262d-n400-20261106b` reached a validation selection score
  (`mean_target_crps`) of 0.400; the ten-epoch run
  `synth-pretrain-5a2262d-n400-e10-20261106` (job 48651082) reached 0.367, with
  the validation loss fluctuating between 1.33 and 1.91 from epoch 1 on. The
  n100 pretrain reached 0.108.

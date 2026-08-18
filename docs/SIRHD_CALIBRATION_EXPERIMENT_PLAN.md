# SIRHD latent-rescaling calibration study

Status: ready for immutable MN5 execution.

## Purpose

Measure whether latent-target scaling restores useful SIRHD gradients during
synthetic pretraining without degrading observation forecasts. The frozen
criteria are in `LATENT_RESCALING_V1_STUDY_CONTRACT.md`.

## Implemented support

- Destination-dated `latent_death_flow_true` replaces cumulative-death loss.
- Latent scaling modes are `none`, `std`, and `asinh`.
- Training writes per-component raw, transformed, and weighted losses.
- The training split creates an immutable `latent_scale_receipt.json`.
- The gradient telemetry sidecar is controlled by
  `pathway_gradient_frequency`.
- Oracle-X0, teacher-forced one-step, and free-rollout diagnostics provide
  latent-pathway evidence.

## Matched arms

| Arm | Config | Latent scaling | Latent weight | Purpose |
| --- | --- | --- | ---: | --- |
| 0 | `configs/experiment/sirhd_latent_rescaling_baseline.yaml` | `none` | 0.0 | Observation-only baseline |
| 1 | `configs/experiment/sirhd_latent_rescaling_unscaled.yaml` | `none` | 0.01 | Unscaled comparison |
| 2 | `configs/experiment/sirhd_latent_rescaling_std.yaml` | `std` | 0.01 | Standard-deviation scaling |
| 3 | `configs/experiment/sirhd_latent_rescaling_asinh.yaml` | `asinh` | 0.01 | Robust scaling for I, H, and death flow |

Every arm uses Normal NLL observation scoring, three epochs,
`pathway_gradient_frequency: 100`, `write_granular_eval: true`,
`gradient_clip_value: 5.0`, and complete latent targets. The latent transform
is the only experimental variable.

## Existing evidence

An earlier post-refactor smoke run found fraction-space imbalance: S and R
losses were about `0.79`, while I, H, and death-flow losses were about
`1.1e-5`, `3.4e-7`, and `1.7e-11`. With unscaled latent weight `0.01`, the
weighted latent loss was about `0.003` versus wastewater NLL about `2333`.
That run did not enable gradient telemetry. The matched arms have not yet
produced their study evidence.

## Required evidence

For each arm, collect:

1. Median combined latent squared-gradient share, per-task gradient norms, and
   cosine with the observation gradient from `training_dynamics.jsonl`.
2. Per-compartment latent MAE/RMSE and horizon metrics from granular evaluation.
3. SIRHD mass error, death-transition error, and physics-residual metrics.
4. Observation R2 and MAE guardrails for wastewater, cases, hospitalizations,
   and deaths.

A scaled arm must satisfy every guardrail in
`LATENT_RESCALING_V1_STUDY_CONTRACT.md` against the unscaled arm. If both pass,
select the stronger gradient-share result, then the lower teacher-forced RMSE,
then `std` as the simpler transform.

## Execution

1. Prepare one immutable MN5 run per matched config. Use the project workflow
   documented in `MN5_SNAKEMAKE.md`; do not reuse or mutate a previous run.
2. Verify the reporting-v1 input receipt and generated
   `latent_scale_receipt.json` for every arm.
3. Run the four arms with identical workflow resources and seeds.
4. Compare only canonical training-dynamics and evaluation artifacts. Record
   the selected arm or the reason every scaled arm failed a guardrail.

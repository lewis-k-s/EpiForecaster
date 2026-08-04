# Latent Rescaling V1 Study Contract

Status: frozen before MN5 submission
Date: 2026-08-05

## Arms

The study compares one complete synthetic epoch for `unscaled_mse`, `std_mse`,
and `asinh_mse`. Each arm uses seed 42, split seed 42, the same ordered windows,
batch size, model initialization, optimizer, scheduler, observation loss, task
weights, precision policy, clipping threshold, checkpoint rule, and canonical
test evaluation. The only scientific change is the latent transform mode.

All arms use the new `latent_flow_v2` processed DVC artifact. An arm is invalid
if its source receipt, container identity, DVC identity, split membership, scale
receipt, number of completed training batches, or terminal evaluation artifacts
are incomplete or do not match the study.

## Primary evidence

Compare the coefficient-weighted latent gradients on the frozen X0 and rate
projection subset. Use the persisted subset hash. The primary signal is the
median sampled `latentgrad_combined_squared_gradient_share`. Also report each
task norm, its cosine with the observation gradient, and all pairwise latent
cosines.

## Guardrails

A scaled arm can pass only when all these conditions hold against `unscaled_mse`:

- The median combined latent squared-gradient share improves by at least 10x.
- The geometric mean absolute log error of the I, H, and death-flow
  prediction-to-target standard-deviation ratios improves by at least 20%.
- The mean teacher-forced one-step RMSE across I, H, and death flow improves by
  at least 10%.
- The mean oracle-X0 RMSE across I, H, and death flow does not degrade by more
  than 2%.
- The mean free-rollout RMSE across I, H, and death flow does not degrade by more
  than 5%.
- S and R free-rollout RMSE do not degrade by more than the larger of 5% relative
  or `1e-4` absolute fraction units.
- The mean observation MAE across cases, hospitalizations, deaths, and wastewater
  does not degrade by more than 5%.
- No finite/nonnegative, conservation, or complete-target contract fails.

All relative comparisons use `max(abs(unscaled), 1e-12)` as the denominator.
The standard-deviation-ratio score excludes a task if its target standard
deviation is at most `1e-12`; the exclusion and eligible count must be reported.

## Selection rule

Reject any arm that fails a guardrail. If neither scaled arm passes, retain the
corrected unscaled objective and use the evidence to plan the next intervention.
If one scaled arm passes, select it. If both pass, select the arm with the larger
median combined latent squared-gradient share. Break a difference smaller than
5% in favor of the arm with the lower mean teacher-forced I/H/death-flow RMSE,
then in favor of `std_mse` because it is the simpler transform.

Do not select an arm from transformed loss magnitude or one qualitative plot.

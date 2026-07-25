# Gradient and GradNorm Interpretation

## Contents

- Terminology
- Total gradients and clipping
- Component gradients
- Adaptive GradNorm
- Gradient snapshots

## Terminology

Keep these evidence types separate:

1. `gradnorm_total_preclip` is the model's total gradient norm before clipping.
2. Component gradient norms describe parameter-group contributions to that norm.
3. Adaptive GradNorm is a task-loss weighting controller with learned task
   weights and target gradient norms.

Do not describe component shares as adaptive GradNorm behavior.

## Total gradients and clipping

Review the full sampled pre-clip series, the configured threshold, and clip
factors.

- Occasional mild clipping can be a normal guardrail.
- Repeated strong clipping means the optimizer repeatedly receives a rescaled
  direction, but it does not by itself identify the source.
- Relate clipping to loss behavior using exact sampled steps.
- Inspect whether clipping is confined to warmup or transitions, clustered in a
  regime, or persistent throughout the run.
- Treat non-finite norms as a failure signal and locate the first affected step.

Do not compare raw norm magnitudes across runs without checking parameterization,
loss weights, accumulation, precision policy, and clipping configuration.

## Component gradients

Use squared-norm shares because squared disjoint component norms are additive.
Confirm group definitions before expecting shares to sum to 100%; unclassified
parameters or overlapping groups can invalidate that expectation.

For every enabled component, review:

- sampling count and cadence;
- median and upper-tail norm;
- zero or near-zero frequency;
- direction of the log-norm trend;
- median and upper-tail squared share;
- regime shifts aligned with loss and curriculum behavior.

A small share is not proof that a component is useless. It can reflect scale,
loss weighting, parameter count, saturation, sparse activation, or the chosen
group boundary. A zero gradient is concerning only when the component was active
and expected to receive supervision.

## Adaptive GradNorm

Only analyze the controller when canonical telemetry includes task weights,
active masks, task losses, task gradient norms, target norms, and controller
loss.

Review:

- weight conservation and configured minimum weights;
- persistent dominance or starvation of a task;
- weight oscillation;
- whether inactive tasks are excluded correctly;
- observed task norms versus controller targets;
- controller overhead and update cadence;
- behavior before and after warmup.

Without those fields, state that adaptive-controller behavior is unavailable.
Console summaries are not a durable substitute.

## Gradient snapshots

When layer-level snapshots exist:

- separate expected zero gradients for inactive heads from unexpected zero
  gradients for active heads;
- report non-finite layers first;
- inspect repeated vanishing/exploding layers across snapshots, not a single
  occurrence in isolation;
- relate snapshot coverage to supervision density;
- use top and bottom norms to choose a narrower investigation, not to rank layer
  importance.

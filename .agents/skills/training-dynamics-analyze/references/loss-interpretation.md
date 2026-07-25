# Loss and Convergence Interpretation

## Contents

- Evidence boundary
- Curve review
- Stability and convergence
- Cross-signal checks
- Safe conclusions

## Evidence boundary

`training_dynamics.jsonl` contains sampled optimizer-step observations, not every
batch. State the sampling cadence and observation count. A sampled training loss
curve can describe optimization behavior but cannot establish generalization.
Only discuss overfitting when canonical validation history is available.

For failed, pruned, running, or non-finite-halted artifacts, interpret the curve
as partial-run evidence. Do not compare its final window directly with a
completed run without matching the step or epoch range.

## Curve review

Inspect both raw sampled loss and the rolling median.

- Use raw observations to locate spikes, discontinuities, and noise.
- Use the rolling median and epoch summaries to describe the underlying trend.
- Report first, middle, and recent behavior rather than only first-versus-last.
- Report the best sampled loss and whether the recent curve retained or lost that
  improvement.
- Inspect the exact neighborhood around robust spikes. Record whether the next
  sampled points recover, remain elevated, or begin a new regime.
- Treat a low recent slope as a plateau candidate only when the recent rolling
  variation is also small. A noisy flat slope is instability, not convergence.

Do not assign meaning to absolute loss values without the persisted loss
configuration, scaling, and component weights.

## Stability and convergence

Use robust window medians and MAD before means and standard deviations. Training
loss is often heavy-tailed because supervision density and target composition
vary by batch.

Describe, without universal thresholds:

- direction and magnitude of the recent trend;
- variation relative to the curve median;
- frequency, magnitude, clustering, and recovery of spikes;
- discontinuities at epoch, curriculum, or learning-rate transitions;
- non-finite observations and the first affected step;
- divergence after the best sampled point.

A decreasing curve with occasional recovered spikes can still be healthy. A
smooth training curve can still generalize poorly. A short run may be useful for
diagnostics while remaining insufficient for a convergence claim.

## Cross-signal checks

At each interesting step, check:

- learning rate and warmup/scheduler phase;
- pre-clip gradient norm and clip factor;
- component gradient shares;
- supervision/curriculum fields when persisted;
- data-loading and model-step duration;
- gradient snapshot or profiler coverage.

Report exact-step coincidence. Do not interpolate sparse component gradients
onto loss observations and then present the result as an exact association.

## Safe conclusions

Prefer language such as:

- “The sampled training curve improved over the observed interval.”
- “The recent rolling median is nearly flat, consistent with a plateau candidate.”
- “Three early spikes recovered by the next sampled observations.”
- “Validation history was not persisted, so generalization cannot be assessed.”

Avoid:

- “The model converged” from a short or training-only run.
- “Gradient clipping caused the spike.”
- “The optimizer is wrong” without a controlled comparison.

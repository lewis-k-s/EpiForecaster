# Current Observation Loss Scaling

This note summarizes how observation losses currently work in EpiForecaster. It
is intended as grounding for research on loss scaling under variable data
sparsity and data quality.

The relevant implementation is in:

- `evaluation/losses.py`: `JointInferenceLoss`
- `training/gradnorm.py`: `GradNormController`
- `training/epiforecaster_trainer.py`: adaptive training and sidecar updates
- `models/configs.py`: `JointLossConfig`

## Observation Heads

The observation objective covers four forecast heads:

```text
ww, hosp, cases, deaths
```

Each head can be disabled through loss config flags. A head is active for a
batch only when its target exists and its effective observed-point count is
positive after hard masking.

For each head `h`, supervision uses:

```text
observed_binary = (observed_mask > 0.5) * finite(target)
n_eff_h = sum(observed_binary)
active_h = n_eff_h > 0
```

If `<head>_min_observed > 0`, series with too few observed points are removed
before `n_eff_h` is computed.

## Per-Head Component Loss

Each active observation head first computes a masked MSE over all observed
points:

```text
L_raw_h = sum_i,t observed_i,t * (prediction_i,t - target_i,t)^2
          / max(sum_i,t observed_i,t, 1)
```

`n_eff_h` is logged for diagnostics and active-head detection. It does not scale
the loss.

Important current behavior: `JointInferenceLoss` returns `ww`, `hosp`, `cases`,
and `deaths` as the raw masked per-head MSE values, plus separate diagnostic
`ww_n_eff`, `hosp_n_eff`, `cases_n_eff`, and `deaths_n_eff` counts.

## Static Loss Mode

Static mode is selected with:

```yaml
training:
  loss:
    joint:
      adaptive_scheme: none
```

In this mode, observation weights are an equal split across active heads:

```text
w_h = obs_weight_sum / count(active heads)  if active_h
w_h = 0                                    otherwise
```

The observation total is:

```text
L_obs_static = sum_h w_h * L_raw_h
```

Consequences:

- The active observation weights sum to `obs_weight_sum`.
- Sparsity affects the estimate variance of `L_raw_h`, not its configured loss
  scale.

The full training loss also includes optional non-observation terms:

```text
L_total = L_obs
        + w_sird_supervision * L_sird_supervision
```

## GradNorm Loss Mode

GradNorm mode is selected with:

```yaml
training:
  loss:
    joint:
      adaptive_scheme: gradnorm
```

The current `JointLossConfig` default is `adaptive_scheme: none`; GradNorm is an
explicit opt-in mode.

GradNorm learns observation weights for the same four heads. The learned
parameters are unconstrained log weights:

```text
raw_weight_h = exp(log_weight_h)
```

For the currently active heads, raw weights are clamped below by
`gradnorm_min_weight`, then normalized to sum to `gradnorm_obs_weight_sum`:

```text
w_h = normalize_active(raw_weight_h, active_h, gradnorm_obs_weight_sum)
```

The default observation budget is:

```text
gradnorm_obs_weight_sum = 0.95
```

During the main training step, cached GradNorm weights are applied to raw masked
per-head components:

```text
L_obs_gradnorm = sum_h w_h * L_raw_h
```

If a head is inactive in the current batch, the trainer zeros its cached weight
and renormalizes the remaining active weights to `obs_weight_sum` before
composing the training loss.

## GradNorm Controller Update

The GradNorm controller is updated in a sidecar pass rather than directly inside
the main optimizer step.

Current defaults:

```text
gradnorm_alpha = 1.5
gradnorm_weight_lr = 1.0e-3
gradnorm_warmup_steps = 50
gradnorm_update_every = 16
gradnorm_ema_decay = 0.9
gradnorm_probe = obs_context
gradnorm_min_weight = 1.0e-3
```

The sidecar update does the following:

1. Runs a forward pass.
2. Computes `ww`, `hosp`, `cases`, and `deaths` components.
3. Stacks those components as the GradNorm task losses.
4. Initializes `L0_h` independently for each active head once warmup has passed.
5. Computes gradient norms against `model_outputs["obs_context"]`.
6. Backpropagates the GradNorm loss into the controller weights only.
7. Refreshes cached weights for future main training steps.

The controller receives the raw masked losses:

```text
loss_for_controller_h = L_raw_h
```

For each active head, the probe gradient norm is:

```text
base_grad_h = || grad_probe(L_raw_h) ||
G_h = w_h * base_grad_h
```

The controller tracks EMA-smoothed losses and gradient norms:

```text
smooth_loss_h = EMA(L_raw_h)
smooth_G_h = EMA(G_h)
```

Training rate is measured against each head's independently initialized
baseline:

```text
rel_train_h = smooth_loss_h / L0_h
rate_h = rel_train_h / mean_active(rel_train)
```

The GradNorm target is:

```text
target_h = mean_active(smooth_G) * rate_h ** gradnorm_alpha
```

The controller objective is:

```text
L_gradnorm = sum_h |G_h - target_h|
```

This objective updates only the GradNorm log weights. The model gradients from
the sidecar pass are cleared afterward.

## Sparsity Handling

Sparsity is handled through hard masking, optional `min_observed` filtering, and
diagnostic `n_eff` counts. It does not change the loss scale.

GradNorm remains an opt-in multi-task balancing method. It is not a sparsity or
observation-quality correction.

## Research-Relevant Baselines

The current system naturally defines these baseline objectives:

```text
Static:
  adaptive_scheme = none
  L_obs = sum_h equal_weight_h * L_raw_h

GradNorm:
  adaptive_scheme = gradnorm
  L_obs = sum_h learned_weight_h * L_raw_h
```

The first case is the current default style in local and production training
configs.

## Current Limitations For Loss-Scaling Research

- GradNorm targets are equal-gradient targets, not observation-quality targets.
- `L0_h` baselines initialize independently when each head first becomes active
  after warmup, which can make training-rate comparisons sensitive to sparse or
  intermittent activation patterns.
- The sidecar forward currently does not pass input masking flags, while the main
  training step does. Under input-ablation experiments, the controller may not
  observe exactly the same forward contract as the main optimizer step.

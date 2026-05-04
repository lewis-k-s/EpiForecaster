# Prior-Aware GradNorm Status

This design note is retired.

EpiForecaster no longer uses effective-support loss scaling. Observation heads
now optimize raw masked MSE over observed points, with `n_eff` retained only as a
diagnostic count and for active-head detection after `min_observed` filtering.

GradNorm remains an explicit opt-in multi-task balancing method. It should not be
used as the mechanism for sparse or noisy observation reliability unless a future
experiment demonstrates that behavior directly against the static baseline.

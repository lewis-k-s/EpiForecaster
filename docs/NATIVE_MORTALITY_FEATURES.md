# Native mortality context

`native.mortality_features_enabled=true` adds five channels after the six
native history channels. Both ABS heads receive these channels. Municipality
and EDAR rows receive zeros in these positions.

1. Expected daily all-cause deaths from the fixed 2017–19 calendar baseline.
2. Mean observed daily all-cause deaths over the previous seven days.
3. Mean observed daily all-cause deaths over the previous fourteen days.
4. Observed-day fraction for the seven-day window.
5. Observed-day fraction for the fourteen-day window.

The first three channels use `log1p(count / ABS graph population * 100000)`.
The last two channels are fractions in `[0, 1]`. Each lag window ends one day
before its input date. Masked days do not contribute to the count or the mean
denominator. An empty window has zero mean and zero coverage. Coverage uses the
full window length, including at the start of the dataset. Feature windows and
lag are recorded in the asset metadata and preprocessing configuration.

The asset retains the parent counts, masks, ages, expected values, and signed
excess deaths. Daily targets and evaluation calculations do not change. The
dataset loader checks the stored features against the causal contract. Input
identity includes the configured mortality asset and the feature flag.

Use `workflow/config/mn5_native_mortality_features.yaml` to build and publish a
new asset through the immutable preprocessing DAG. Import its run-owned DVC
pointer as `data/processed/native_support/obs_deaths_abs_mortality_features_v1.zarr.dvc`.
The draft `workflow/config/drafts/mn5_native_mortality_10ep.yaml` selects
that asset and the V3 clinical dataset. Publish and import the named mortality
DVC pointer before promoting the draft to `workflow/config/`. It retains the
four heads, 60-day context, 28-day forecast horizon, seed 42, fold 0, and broad purged validation
policy from the plateau report. The new five input columns change model
initialization, and the V3 input change is part of this experiment.

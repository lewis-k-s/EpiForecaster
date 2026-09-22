# Native Stage-4 matched comparisons

`base.yaml` and `matrix.yaml` define the frozen clinical comparison contract.
The matrix has 16 conditions for each fold and seed. Run the three phases in
order: encoder, mobility, then OD context. Do not run the full matrix as an
uncontrolled local sweep.

Substantive execution is MN5-only. This slice freezes the experiment contract
but does not authorize dispatch. Before any run, version the native-support
artifacts, add their identities to the workflow input manifest, and extend the
immutable workflow to accept the native trainer's checkpoint and
training-dynamics artifacts. Direct cluster CLI runs are prohibited. After
that workflow gate is implemented, render each condition with these overrides:

```text
native.spatial_encoder=<pooled|hetero>
native.mobility_enabled=<false|true>
training.loss.joint.observation_loss=<nll|crps>
native.od_source=<mitma|municipal>
native.fold_index=<0..9>
training.seed=<52|53>
output.experiment_name=<output_name_template>
```

The mobility-off condition zeroes the fixed-width OD context column and removes
OD graph messages without loading an OD artifact. Native histories, targets,
temporal covariates, source folds, model
hyperparameters, optimizer settings, and confirmation-period protection stay
fixed. The native trainer writes the final best-state validation granular CSV,
aggregate metrics, and metadata needed to audit capacity, optimizer exposure,
and elapsed compute.

After all condition artifacts are materialized, run:

```bash
uv run python scripts/analysis/native_stage4_comparison.py \
  --runs-root outputs/training/native_stage4 \
  --historical-root outputs/loss_research/loss-type-full-coverage-v2 \
  --output-dir outputs/reports/native_stage4
```

The analysis requires the complete 16-condition matrix for every fold-seed it
finds. It uses common source/date/horizon rows for each direct comparison and
keeps the historical municipal diagnostic in a separate output table.

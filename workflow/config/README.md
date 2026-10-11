# Immutable workflow templates

Use the general templates for training, evaluation, HPO, ablation, and
cross-validation. `mn5_train.yaml` is the production interface.
`mn5_train_smoke.yaml` is the bounded MN5 acceptance interface.
`configs/train_epifor_smoke.yaml` remains bounded CPU acceptance only.

Both model configs extend `configs/native_base.yaml`. Workflow templates share
native input lists and ACC allocation fields from `profiles/`. Bases are paths
relative to the source root. Mappings merge; ordinary lists replace. Dotted-key
`overrides` lists merge by key, with the later assignment taking precedence.
Base cycles, paths outside the source root, and missing variants are errors.

## Preparation

Select a named variant before preparing the immutable run:

```bash
bash scripts/workflow/mn5_workflow.sh prepare <run-id> \
  --config workflow/config/mn5_native_performance.yaml --variant baseline
```

Preparation can change workflow fields with `--override`. For example, shorten
the allocation without creating a second performance config:

```bash
bash scripts/workflow/mn5_workflow.sh prepare <run-id> \
  --config workflow/config/mn5_native_performance.yaml --variant compile_scores \
  --override resources.allocation.runtime=8
```

Training dotlist entries remain in `training.overrides`. To change them from
preparation, assign the complete list to that workflow field. Model and workflow
schemas validate the final config. Preparation saves one complete YAML file,
the selected variant, preparation overrides, and the source snapshot identity.
Dispatch accepts no further settings. Existing immutable runs are not changed.

## Collections

| File | Variants and purpose |
| --- | --- |
| `mn5_native_performance.yaml` | `baseline`, `blas`, `cache`, `blas_cache`, `routing`, `combined`, `compile_sir`, `compile_scores`, `all`, `all_b64`, `all_causal`, `direct` |
| `mn5_native_acceptance.yaml` | ABS/vaccination acceptance and wastewater worker/batch diagnostics; select a key from `variants` |
| `mn5_native_direct.yaml` | `full` trains the causal direct decoder; `evaluate` reproduces evaluation of its pinned producer checkpoint |
| `reproductions/native_abs_excess.yaml` | `resume_b16`, `resume_b32`, and `evaluate` retain explicit historical checkpoint paths |
| `mn5_native_wastewater_history_repair.yaml` | Rebuild causal histories from one pinned parent; preserve raw observations and publish a new asset |

The old confirm-control, CPU-short, and compile-scores-short files used the same
settings as `baseline`, `blas_cache`, and `compile_scores`, with shorter runtimes.
Use an explicit runtime override to repeat those budgets.

`all` uses the original wastewater input. `all_causal` and `direct` use the
repaired input and carry its repair provenance. Compare `all_causal` with
`direct` to isolate the decoder change. A comparison with `baseline` also
changes the input; it does not isolate performance flags.

Pinned checkpoint recipes belong to reproduction work. Check each source path
before use. New runs should use a general template and an explicit checkpoint.
The generic evaluation template requires its checkpoint to be filled before
preparation.

## Full native decoder and mobility coverage

`mn5_native_decoder_mobility_10ep_allfolds.yaml` runs SIR and direct decoding
with mobility on and off on the V3 clinical data and published mortality
features. The campaign evaluates all ten native source folds with model seed
42: four conditions times ten folds gives 40 runs. Each run uses ten full
epochs and final validation/test evaluation.

Native source membership uses the version-4 deterministic assignment policy
and split seed 42. Fold `r` is the test fold, `(r + 1) % 10` is the validation
fold, and the remaining eight folds are training folds. Thus every eligible
source is held out for test once and validation once per condition. Calendar
purging and eligibility still apply to query targets.

The allocation uses four GPU workers, 20 CPUs per worker, and an eight-hour
budget. Its final 30 minutes are reserved for controlled closeout. The exact
40-identity matrix and finite held-out metrics are required for completion.

## Drafts with unpublished inputs

`drafts/mn5_native_mortality_10ep.yaml` records the proposed mortality feature
experiment. Its mortality DVC pointer is not yet present. Build and publish the
asset with `mn5_native_mortality_features.yaml`, import the pointer, then promote
the draft to this directory. The mortality tests check the draft's config; the
standard template tests require every declared input pointer to exist.

## Retired interfaces

Municipal training, synthetic model pretraining, latent-rescaling training
arms, and the pooled-versus-heterogeneous Stage-4 matrix are retired here.
Municipal analysis commands require an explicit config from the municipal
worktree. Historical comparison scripts can still read old result sets.

The native HPO search uses native OD thresholds and heterogeneous encoder
widths. It does not sample unused municipal data or GNN settings. Native study
and campaign IDs are separate from the old municipal IDs.

# MN5 native production configuration

Do not run these configs locally. Use the immutable MN5 Snakemake interface
in `scripts/workflow/mn5_workflow.sh` and `docs/MN5_SNAKEMAKE.md`.

`train_epifor_mn5_full.yaml` is the production native heterogeneous configuration.
`../train_epifor_smoke.yaml` is the bounded local acceptance configuration.
Both use municipality cases, weekly ABS admissions, daily ABS all-cause deaths,
and EDAR assays. Native graph and clinical inputs must be staged from DVC.

The DAG derives purged, representative source sets from processed inputs,
publishes the Parquet assignments and small manifest to DVC, and then trains.
It runs final evaluation as a separate step. Changed inputs or eligibility
settings invalidate the split asset. See `EPIFORECASTER.md` for the contracts.

The municipal trainer, pooled encoder, and old source-membership JSON routes
are not supported in this worktree. Legacy production experiments belong to
the separate legacy worktree.

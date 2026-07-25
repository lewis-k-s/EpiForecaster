# Training Evidence Provenance and Lifecycle

## Evidence tiers

Treat each tier separately:

| Tier | Identity | Default lifecycle |
| --- | --- | --- |
| Source snapshot and workflow configuration | SHA-256 identities in the immutable run manifest | Durable |
| Dataset and embedding inputs | DVC pointer path plus DVC object hash | Durable |
| Snakemake dependency contracts | Stage fingerprint plus contract SHA-256 | Run durable |
| Training dynamics, manifests, and profiler traces | Per-file SHA-256 | Run scoped |
| Retained training evidence pack | DVC directory pointer plus constituent SHA-256 values | Explicitly retained |
| HTML, CSV, and PNG report outputs | Reproducible from the evidence bundle | Replaceable |

Do not call a training artifact DVC-tracked merely because it has a SHA-256.
Use `dvc_tracking=not_recorded` until a DVC pointer for an evidence pack exists.

## Report provenance

Use `training_provenance.json` when present. It is emitted by the Snakemake
training workflow and travels with the model directory. Otherwise, analyze the
entity inside its immutable run or pass the exact `manifest.yaml`.

The report must identify:

- run ID and immutable run-manifest hash;
- source-snapshot, configuration, container, and workflow-tooling identities;
- each DVC input pointer, object hash, role, and materialization reconciliation;
- each Snakemake stage dependency fingerprint available to the entity;
- SHA-256 and byte size of the dynamics, manifest, and profiler artifacts;
- whether training evidence is only run-scoped or explicitly DVC-retained.

Do not execute `dvc pull` merely to render provenance. Pointer metadata and run
receipts are sufficient unless the report also needs the payload.

## Retention policy

Do not DVC-publish every trial, plot, or HTML rendition.

Use run-scoped hashed evidence for failed, pruned, exploratory, and routine
intermediate runs. Retain the immutable run while it remains actionable.

Create one compact DVC evidence pack when a run is selected for comparison,
publication, audit, promotion, or long-lived reporting. Pack the canonical
dynamics JSONL and manifest, profiler traces, training manifest, and portable
provenance record. Keep reports outside that pack because they can be
regenerated.

Expire evidence by removing its retained pointer or immutable run as one
explicit lifecycle action. Do not overwrite files inside an immutable run.
Shared-remote garbage collection must account for every retained pointer across
run and promotion roots before deleting cache objects.

# MN5 DVC pilot

This pilot uses DVC for immutable dataset identity and the existing MN5 dispatch
stack for execution. Each local worktree is synced into a different remote
project directory, while DVC materializes the same read-only cache objects in
each workspace.

## Remote layout

```text
/home/bsc/bsc008913/
  epiforecaster-dvc/
    storage/                 # DVC NFS remote
    cache/                   # shared protected DVC cache
  epiforecaster-runs/
    <run-id>/project/        # isolated code, data links, logs, and outputs
```

The legacy `/home/bsc/bsc008913/EpiForecaster` checkout remains the shared
Python environment during the pilot. It is not used as a code or output
workspace by pilot jobs.

## Pilot datasets

| Role | Workspace path | DVC identity | Files | Bytes |
| --- | --- | --- | ---: | ---: |
| Raw mobility | `data/files/mobility.zarr` | `f3d97d9e1d9128436f8bf498220818eb.dir` | 7,673 | 713,532,624 |
| Canonical processed | `data/processed/real_with_id.zarr` | `1e695877f8565c40dcec4c214b0813b8.dir` | 3,587 | 201,108,907 |

DVC treats each Zarr store as a directory artifact. The smoke job additionally
opens both stores with `xarray.open_zarr()` and records their dimensions and
variables.

## Setup

Install the locked development environment locally and in the legacy MN5
environment:

```bash
uv sync --frozen
```

The committed DVC remote points at the NFS storage directory. Each isolated
remote workspace receives a local cache configuration during `prepare`:

```ini
[cache]
    dir = /home/bsc/bsc008913/epiforecaster-dvc/cache
    shared = group
    type = symlink
```

Do not run `dvc gc` against the shared cache while any worktree or job may still
reference it. Garbage collection is deliberately outside the pilot.

## Prepare and submit two worktrees

Run these commands from each local worktree, choosing distinct IDs:

```bash
bash scripts/cluster/mn5_run.sh prepare dvc-pilot-a
bash scripts/cluster/mn5_run.sh submit dvc-pilot-a dvc_smoke
```

```bash
bash scripts/cluster/mn5_run.sh prepare dvc-pilot-b
bash scripts/cluster/mn5_run.sh submit dvc-pilot-b dvc_smoke
```

The submit command prints the Slurm job ID. Check each job with its matching
workspace:

```bash
bash scripts/cluster/mn5_run.sh status dvc-pilot-a <job-id-a>
bash scripts/cluster/mn5_run.sh status dvc-pilot-b <job-id-b>
```

After both jobs complete, verify dataset identity and workspace isolation:

```bash
bash scripts/cluster/mn5_run.sh verify dvc-pilot-a dvc-pilot-b
```

The verifier requires:

- distinct remote project roots and run IDs;
- the exact two DVC hashes listed above;
- successful Xarray inspection in both jobs;
- links into the shared DVC cache in both workspaces;
- one workspace-local canary per processed Zarr directory;
- no canary or smoke output belonging to the other run.

## Pilot decision gate

Keep the current dispatch CLI if this test proves that DVC pull, cache linking,
and run-directory isolation are reliable on MN5. Consider Snakemake only if the
next requirement is DAG orchestration across preprocessing, training,
evaluation, and dependent job arrays; it is not needed merely to version the
datasets or isolate worktrees.

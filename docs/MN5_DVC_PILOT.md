# MN5 DVC pilot

This pilot uses DVC for immutable dataset identity and the existing MN5 dispatch
stack for execution. Each local worktree is synced into a different remote
project directory, while DVC materializes the same read-only cache objects in
each workspace.

First-time cache population is guarded by
`cache/.dvc-pull.lock`. DVC protects cache objects as read-only, so two
simultaneous writes of the same missing object can otherwise race on NFS. The
lock covers `dvc pull` during preparation; it does not serialize submitted
jobs or their isolated workspaces.

## Remote layout

```text
/gpfs/projects/bsc08/shared_projects/MePreCiSa/EpiForecaster-DVC/
  storage/                   # shared DVC NFS remote
  cache/                     # shared protected DVC cache

/home/bsc/bsc008913/
  epiforecaster-dvc/
    tooling/                 # isolated DVC virtual environment
  epiforecaster-runs/
    <run-id>/project/        # isolated code, data links, logs, and outputs
```

The legacy `/home/bsc/bsc008913/EpiForecaster` checkout supplies the existing
scientific Python environment during the pilot. DVC itself has an isolated
environment under `epiforecaster-dvc/tooling`; the legacy environment is not
modified, and the legacy checkout is not used as a code or output workspace by
pilot jobs.

## Pilot datasets

| Role | Workspace path | DVC identity | Files | Bytes |
| --- | --- | --- | ---: | ---: |
| Raw mobility | `data/files/mobility.zarr` | `f3d97d9e1d9128436f8bf498220818eb.dir` | 7,673 | 713,532,624 |
| Canonical processed | `data/processed/real_with_id.zarr` | `1e695877f8565c40dcec4c214b0813b8.dir` | 3,587 | 201,108,907 |

DVC treats each Zarr store as a directory artifact. The smoke job additionally
opens both stores with `xarray.open_zarr()` and records their dimensions and
variables.

## Setup

Install the locked development environment locally:

```bash
uv sync --frozen
```

Provision the isolated DVC CLI once. MN5 login nodes do not have outbound
PyPI access, so resolve a Linux wheelhouse locally, transfer it through `dt`,
and install from it offline:

```bash
python -m venv /home/bsc/bsc008913/epiforecaster-dvc/tooling
uv pip install \
  --offline \
  --find-links /home/bsc/bsc008913/epiforecaster-dvc/wheelhouse \
  --python /home/bsc/bsc008913/epiforecaster-dvc/tooling/bin/python \
  'dvc==3.67.1'
```

The committed DVC remote points at the NFS storage directory. Each isolated
remote workspace receives a local cache configuration during `prepare`:

```ini
[cache]
    dir = /gpfs/projects/bsc08/shared_projects/MePreCiSa/EpiForecaster-DVC/cache
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

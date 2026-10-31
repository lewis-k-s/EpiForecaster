# MN5 runtime image build

`build_sif_on_mn5.sh` is the sole operator-facing runtime-image build command.
It prepares locked dependencies on the network-enabled MN5 login and submits
the offline assembly and GPU validation step through Slurm.

The other files in this directory are internal stages:

- `fetch_runtime_deps_on_wifi.sh` pulls the pinned base image and materializes
  the frozen `runtime` dependency group using a dependency-specific node-local
  cache.
- `assemble_sif_on_mn5.sbatch` assembles, validates, and atomically publishes
  the immutable SIF from an MN5 compute node.

The SIF contains slow-moving Linux, Python, native, scientific, and DVC
dependencies. EpiForecaster source, Snakefiles, scientific configuration, data,
checkpoints, and results remain outside the image and retain independent
identities.

Scientific ranks are independent Slurm `srun` workers. The current workflow
does not use MPI or distributed-data-parallel training.

Runtime jobs execute the verified SIF directly from shared GPFS. Temporary
files and selected DVC inputs may stage to node-local scratch, but the SIF is
not copied there. Keep that boundary until a benchmark shows that allocation-
level or per-node SIF staging improves total runtime.

Use:

```bash
bash containers/runtime/build_sif_on_mn5.sh --dry-run
bash containers/runtime/build_sif_on_mn5.sh
```

The dependency key includes the pinned base image, `uv.lock`, and both internal
build stages. Any relevant change creates a new immutable request and output;
existing SIFs are not overwritten.

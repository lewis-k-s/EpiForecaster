#!/bin/bash
#SBATCH --job-name=epi-snakemake
#SBATCH --account=bsc08
#SBATCH --partition=acc
#SBATCH --qos=acc_bscls
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=07:00:00

# Run Snakemake as the control plane inside one homogeneous MN5 allocation.
# Scientific rules launch explicit srun steps and own the allocated GPUs.

set -euo pipefail

# -----------------------------------------------------------------------------
# Immutable run and allocation contract
# -----------------------------------------------------------------------------

RUN_ROOT="${1:?run root is required}"
CONTAINER_SIF="${2:?container SIF is required}"
EXPECTED_TASKS="${3:?expected task count is required}"
EXPECTED_CPUS_PER_TASK="${4:?expected CPUs per task is required}"
shift 4

# Reject mismatches between the sealed run config and the actual allocation.
validate_allocation() {
  local allocated_nodes

  test -d "$RUN_ROOT/source"
  test -r "$RUN_ROOT/config.yaml"
  test -f "$CONTAINER_SIF"
  allocated_nodes="${SLURM_JOB_NUM_NODES:-${SLURM_STEP_NUM_NODES:-${SLURM_NNODES:-}}}"

  if [ "$allocated_nodes" != "1" ] \
    || [ "${SLURM_NTASKS:-}" != "$EXPECTED_TASKS" ] \
    || [ "${SLURM_CPUS_PER_TASK:-}" != "$EXPECTED_CPUS_PER_TASK" ]; then
    echo "Expected one MN5 node with $EXPECTED_TASKS tasks and $EXPECTED_CPUS_PER_TASK CPUs per task" >&2
    return 2
  fi
}

validate_allocation

# Load the site module only when Singularity is absent from the batch PATH.
load_singularity() {
  local profile_script candidate loaded=0

  if ! command -v module >/dev/null 2>&1; then
    set +u
    for profile_script in \
      /etc/profile.d/00-bsc.sh \
      /etc/profile.d/01-module.sh \
      /apps/modules/LMOD/8.7/lmod/8.7/init/profile; do
      if [ -r "$profile_script" ]; then
        # shellcheck disable=SC1090
        source "$profile_script"
      fi
      if command -v module >/dev/null 2>&1; then
        break
      fi
    done
    set -u
  fi

  if command -v singularity >/dev/null 2>&1; then
    return 0
  fi

  module load EB/apps >/dev/null 2>&1 || true
  for candidate in \
    "${SINGULARITY_MODULE:-singularity/4.1.5}" \
    "Singularity/4.1.5" \
    "SingularityCE/4.1.5"; do
    if module load "$candidate" >/dev/null 2>&1; then
      loaded=1
      break
    fi
  done

  if [ "$loaded" -ne 1 ]; then
    echo "Unable to load Singularity 4.1.5 module" >&2
    return 1
  fi
}

load_singularity

# Only one coordinator may mutate a run's Snakemake metadata at a time.
exec 9>"$RUN_ROOT/.state/coordinator.run.lock"
if ! flock -n 9; then
  echo "A Snakemake coordinator already owns this run" >&2
  exit 1
fi

# Persist scheduler identities before Snakemake begins constructing the DAG.
execution_id="${SLURM_JOB_ID:?SLURM_JOB_ID is required}_0"
printf '%s\n' "$SLURM_JOB_ID" >"$RUN_ROOT/.state/coordinator.job_id"
printf '%s\n' "$execution_id" >"$RUN_ROOT/.state/coordinator.execution_id"

container_args=(
  exec
  --cleanenv
  --containall
  --bind "$RUN_ROOT:$RUN_ROOT:rw"
  --bind /usr/bin/srun:/usr/bin/srun:ro
  --bind /usr/lib64/slurm:/usr/lib64/slurm:ro
  --bind /usr/lib64/libmunge.so.2.0.1:/usr/lib64/libmunge.so.2.0.1:ro
  --bind /usr/lib64/libmunge.so.2:/usr/lib64/libmunge.so.2:ro
  --bind /opt/.snoopy/lib:/opt/.snoopy/lib:ro
  --bind /opt/EAR/current:/opt/EAR/current:ro
  --bind /etc/slurm:/etc/slurm:ro
  --bind /etc/passwd:/etc/passwd:ro
  --bind /etc/group:/etc/group:ro
  --bind /run/munge:/run/munge:ro
  --env "PATH=/opt/epiforecaster/.venv/bin:/usr/local/bin:/usr/bin:/bin"
  --env "PYTHONDONTWRITEBYTECODE=1"
  --env "PYTHONPATH=$RUN_ROOT/source"
  --env "XDG_CACHE_HOME=$RUN_ROOT/.cache"
  --env "TMPDIR=/tmp"
  --pwd "$RUN_ROOT"
)

# Clean environments do not automatically preserve the allocation identity
# needed by the host srun client exposed above.
while IFS= read -r variable_name; do
  container_args+=(--env "$variable_name=${!variable_name}")
done < <(compgen -A variable SLURM_)

# Replace the batch shell so Snakemake's status is the Slurm job status.
exec singularity "${container_args[@]}" "$CONTAINER_SIF" /usr/local/bin/snakemake \
  --snakefile "$RUN_ROOT/source/workflow/Snakefile" \
  --configfile "$RUN_ROOT/config.yaml" \
  --workflow-profile "$RUN_ROOT/source/workflow/profiles/mn5" \
  --directory "$RUN_ROOT" \
  --runtime-source-cache-path "$RUN_ROOT/.snakemake/source-cache" \
  "$@"

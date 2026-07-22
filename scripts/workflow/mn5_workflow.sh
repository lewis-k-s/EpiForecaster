#!/bin/bash

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
CONF="${CLUSTER_CONF:-$REPO_ROOT/.cluster/mn5.conf}"
# shellcheck disable=SC1090
source "$CONF"

COMPUTE_HOST="${COMPUTE_HOST:?COMPUTE_HOST must be configured}"
TRANSFER_HOST="${TRANSFER_HOST:?TRANSFER_HOST must be configured}"
CONTAINER_NETWORK_HOST="${CONTAINER_NETWORK_HOST:-wifi-acc}"
CONTAINER_NETWORK_HOSTNAME="${CONTAINER_NETWORK_HOSTNAME:-}"
REMOTE_WORKFLOW_RUN_ROOT="${REMOTE_WORKFLOW_RUN_ROOT:?REMOTE_WORKFLOW_RUN_ROOT must be configured}"
REMOTE_WORKFLOW_TOOLING_ROOT="${REMOTE_WORKFLOW_TOOLING_ROOT:?REMOTE_WORKFLOW_TOOLING_ROOT must be configured}"
REMOTE_CONTAINER_SIF="${REMOTE_CONTAINER_SIF:?REMOTE_CONTAINER_SIF must be configured}"
DVC_REMOTE_PATH="${DVC_REMOTE_PATH:?DVC_REMOTE_PATH must be configured}"
DVC_CACHE_PATH="${DVC_CACHE_PATH:?DVC_CACHE_PATH must be configured}"
TEMPLATE="${WORKFLOW_CONFIG_TEMPLATE:-$REPO_ROOT/workflow/config/mn5_train.yaml}"
VPN_HELPER="${WORKVPN_HELPER:-/Users/lewis/.codex/skills/workvpn/vpn}"
SNAKEMAKE_EXPECTED_VERSION="9.23.1"
TOOLING_KEY=""
REMOTE_TOOLING_REQUEST=""
REMOTE_TOOLING_ENV=""
REMOTE_SNAKEMAKE_BIN=""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") tooling
  $(basename "$0") prepare [run-id] [--config TEMPLATE]
  $(basename "$0") dry-run <run-id> [snakemake-args...]
  $(basename "$0") run <run-id> [snakemake-args...]
  $(basename "$0") status <run-id>
  $(basename "$0") unlock <run-id>
  $(basename "$0") path <run-id>

prepare snapshots this worktree into an atomically reserved project-GPFS run.
The default template is workflow/config/mn5_train.yaml; set
WORKFLOW_CONFIG_TEMPLATE to select HPO, ablation, or evaluation configuration.
run submits the one-node ACC allocation declared by that immutable config.
EOF
}

validate_run_id() {
  local run_id="$1"
  if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid run id '$run_id'" >&2
    return 2
  fi
}

generate_run_id() {
  local branch short_sha timestamp
  branch="$(git branch --show-current)"
  branch="${branch:-detached}"
  branch="$(printf '%s' "$branch" | tr '/ :' '---' | tr -cd 'A-Za-z0-9._-')"
  short_sha="$(git rev-parse --short=10 HEAD)"
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  printf '%s-%s-%s-%s\n' "$branch" "$short_sha" "$timestamp" "$$-$RANDOM"
}

run_root() {
  printf '%s/%s\n' "$REMOTE_WORKFLOW_RUN_ROOT" "$1"
}

shell_quote() {
  printf '%q' "$1"
}

remote_bash_on() {
  local host="$1"
  local command="$2"
  local -a ssh_options=()
  if [ "$host" = "$CONTAINER_NETWORK_HOST" ] && \
     [ -n "$CONTAINER_NETWORK_HOSTNAME" ]; then
    ssh_options+=(-o "HostName=$CONTAINER_NETWORK_HOSTNAME")
  fi
  ssh "${ssh_options[@]}" "$host" "bash -c $(shell_quote "$command")"
}

remote_bash() {
  remote_bash_on "$COMPUTE_HOST" "$1"
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

set_tooling_identity() {
  if [ -n "$TOOLING_KEY" ]; then
    return 0
  fi
  TOOLING_KEY="$(PYTHONDONTWRITEBYTECODE=1 python3 -m scripts.workflow.tooling_identity --lock-file "$REPO_ROOT/uv.lock")"
  REMOTE_TOOLING_REQUEST="$REMOTE_WORKFLOW_TOOLING_ROOT/requests/$TOOLING_KEY"
  REMOTE_TOOLING_ENV="$REMOTE_WORKFLOW_TOOLING_ROOT/envs/$TOOLING_KEY"
  REMOTE_SNAKEMAKE_BIN="$REMOTE_TOOLING_ENV/bin/snakemake"
}

remote_tooling_state() {
  local marker probe_code
  marker="$REMOTE_TOOLING_REQUEST/ready"
  probe_code="import importlib.metadata as m; d=m.distribution('snakemake'); print(d.version); print(d.locate_file(''))"
  remote_bash "
if test -x $(shell_quote "$REMOTE_SNAKEMAKE_BIN") && test -f $(shell_quote "$marker"); then
  installed=\$($(shell_quote "$REMOTE_TOOLING_ENV/bin/python") -c $(shell_quote "$probe_code"))
  installed_version=\$(printf '%s\\n' \"\$installed\" | sed -n '1p')
  installed_location=\$(printf '%s\\n' \"\$installed\" | sed -n '2p')
  if grep -Fxq $(shell_quote "tooling_key=$TOOLING_KEY") $(shell_quote "$marker") && \\
     grep -Fxq $(shell_quote "snakemake_version=$SNAKEMAKE_EXPECTED_VERSION") $(shell_quote "$marker") && \\
     grep -Fxq \"snakemake_location=\$installed_location\" $(shell_quote "$marker") && \\
     test \"\$installed_version\" = $(shell_quote "$SNAKEMAKE_EXPECTED_VERSION"); then
    printf ready
    exit 0
  fi
fi
printf missing"
}

ensure_tooling() {
  local context marker probe_code remote_state sync_command
  set_tooling_identity
  uv lock --check --project "$REPO_ROOT" >/dev/null
  context="$REMOTE_TOOLING_REQUEST/context"
  marker="$REMOTE_TOOLING_REQUEST/ready"
  probe_code="import importlib.metadata as m; d=m.distribution('snakemake'); print(d.version); print(d.locate_file(''))"
  remote_state="$(remote_tooling_state)"
  if [ "$remote_state" = "ready" ]; then
    echo "Locked Snakemake tooling is ready: $REMOTE_TOOLING_ENV"
    return 0
  fi

  remote_bash "mkdir -p $(shell_quote "$context") $(shell_quote "$REMOTE_WORKFLOW_TOOLING_ROOT/envs") $(shell_quote "$REMOTE_WORKFLOW_TOOLING_ROOT/cache") $(shell_quote "$REMOTE_WORKFLOW_TOOLING_ROOT/python")"
  rsync -a "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/uv.lock" \
    "$TRANSFER_HOST:$context/"
  if [ -f "$REPO_ROOT/README.md" ]; then
    rsync -a "$REPO_ROOT/README.md" "$TRANSFER_HOST:$context/"
  fi
  if [ -x "$VPN_HELPER" ]; then
    "$VPN_HELPER" wait 60
  fi

  sync_command="set -euo pipefail
export PATH=\"\$HOME/.local/bin:\$PATH\"
request=$(shell_quote "$REMOTE_TOOLING_REQUEST")
environment=$(shell_quote "$REMOTE_TOOLING_ENV")
context=$(shell_quote "$context")
marker=$(shell_quote "$marker")
mkdir -p \"\$request\" \"\$environment\" $(shell_quote "$REMOTE_WORKFLOW_TOOLING_ROOT/cache") $(shell_quote "$REMOTE_WORKFLOW_TOOLING_ROOT/python")
exec 9>\"\$request/sync.lock\"
flock 9
if test -x \"\$environment/bin/snakemake\" && test -f \"\$marker\"; then
  installed=\$(\"\$environment/bin/python\" -c $(shell_quote "$probe_code"))
  installed_version=\$(printf '%s\\n' \"\$installed\" | sed -n '1p')
  if grep -Fxq $(shell_quote "tooling_key=$TOOLING_KEY") \"\$marker\" && \\
     grep -Fxq $(shell_quote "snakemake_version=$SNAKEMAKE_EXPECTED_VERSION") \"\$marker\" && \\
     test \"\$installed_version\" = $(shell_quote "$SNAKEMAKE_EXPECTED_VERSION"); then exit 0; fi
fi
chmod -R u+w \"\$environment\"
command -v uv >/dev/null
UV_PROJECT_ENVIRONMENT=\"\$environment\" \\
UV_CACHE_DIR=$(shell_quote "$REMOTE_WORKFLOW_TOOLING_ROOT/cache") \\
UV_PYTHON_INSTALL_DIR=$(shell_quote "$REMOTE_WORKFLOW_TOOLING_ROOT/python") \\
UV_LINK_MODE=copy \\
uv sync \\
  --project \"\$context\" \\
  --frozen \\
  --only-group workflow \\
  --no-install-project \\
  --python 3.11
PYTHONDONTWRITEBYTECODE=1 \"\$environment/bin/snakemake\" --version
installed=\$(\"\$environment/bin/python\" -c $(shell_quote "$probe_code"))
installed_version=\$(printf '%s\\n' \"\$installed\" | sed -n '1p')
installed_location=\$(printf '%s\\n' \"\$installed\" | sed -n '2p')
test \"\$installed_version\" = $(shell_quote "$SNAKEMAKE_EXPECTED_VERSION")
test -n \"\$installed_location\"
chmod -R a-w \"\$environment\"
temporary=\"\$marker.$$\"
printf '%s\n' \\
  tooling_key=$(shell_quote "$TOOLING_KEY") \\
  snakemake_version=\"\$installed_version\" \\
  snakemake_location=\"\$installed_location\" \\
  pyproject_sha256=$(shell_quote "$(sha256_file "$REPO_ROOT/pyproject.toml")") \\
  uv_lock_sha256=$(shell_quote "$(sha256_file "$REPO_ROOT/uv.lock")") \\
  uv_version=\"\$(uv --version)\" > \"\$temporary\"
chmod 0444 \"\$temporary\"
mv \"\$temporary\" \"\$marker\""
  remote_bash_on "$CONTAINER_NETWORK_HOST" "$sync_command"
  remote_state="$(remote_tooling_state)"
  if [ "$remote_state" != "ready" ]; then
    echo "Remote tooling failed provenance verification: $REMOTE_TOOLING_ENV" >&2
    return 1
  fi
  echo "Locked Snakemake tooling prepared: $REMOTE_TOOLING_ENV"
}

snakemake_command() {
  local run_id="$1"
  shift
  local root command
  set_tooling_identity
  root="$(run_root "$run_id")"
  command="cd $(shell_quote "$root")
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$(shell_quote "$root/source")
export XDG_CACHE_HOME=$(shell_quote "$root/.cache")
$(shell_quote "$REMOTE_SNAKEMAKE_BIN") \\
  --snakefile $(shell_quote "$root/source/workflow/Snakefile") \\
  --configfile $(shell_quote "$root/config.yaml") \\
  --workflow-profile $(shell_quote "$root/source/workflow/profiles/mn5") \\
  --directory $(shell_quote "$root") \\
  --runtime-source-cache-path $(shell_quote "$root/.snakemake/source-cache")"
  for arg in "$@"; do
    command+=" $(shell_quote "$arg")"
  done
  printf '%s\n' "$command"
}

cleanup_failed_run() {
  local root="$1"
  local token="$2"
  remote_bash "
if test -f $(shell_quote "$root/.state/preparing") && test \"\$(cat $(shell_quote "$root/.state/preparing"))\" = $(shell_quote "$token"); then
  chmod -R u+w $(shell_quote "$root") 2>/dev/null || true
  rm -rf -- $(shell_quote "$root")
fi" || true
}

prepare_run() {
  local run_id="${1:-}"
  local root resolved_sif tmp_base tmp_dir snapshot rendered file_list token
  if [ -z "$run_id" ]; then
    run_id="$(generate_run_id)"
  fi
  validate_run_id "$run_id"
  root="$(run_root "$run_id")"

  ensure_tooling
  resolved_sif="$(remote_bash "readlink -f $(shell_quote "$REMOTE_CONTAINER_SIF")")"
  remote_bash "set -euo pipefail
test -f $(shell_quote "$resolved_sif")
test -f $(shell_quote "$resolved_sif.sha256")
cd $(shell_quote "$(dirname "$resolved_sif")")
sha256sum -c $(shell_quote "$(basename "$resolved_sif.sha256")")"

  tmp_base="${HUBDIR:-/Volumes/HUBSSD}/tmp/epiforecaster-workflow"
  mkdir -p "$tmp_base"
  tmp_dir="$(mktemp -d "$tmp_base/run.XXXXXX")"
  snapshot="$tmp_dir/source"
  rendered="$tmp_dir/config.yaml"
  file_list="$tmp_dir/source-files.bin"

  if ! uv run --no-sync python -m scripts.workflow.run_config render \
    --template "$TEMPLATE" \
    --output "$rendered" \
    --file-list "$file_list" \
    --snapshot-root "$snapshot" \
    --repo-root "$REPO_ROOT" \
    --run-id "$run_id" \
    --run-root "$root" \
    --container-sif "$resolved_sif" \
    --dvc-remote "$DVC_REMOTE_PATH" \
    --dvc-cache "$DVC_CACHE_PATH" \
    --workflow-tooling-key "$TOOLING_KEY" \
    --snakemake-version "$SNAKEMAKE_EXPECTED_VERSION" \
    --snakemake-bin "$REMOTE_SNAKEMAKE_BIN"; then
    rm -rf -- "$tmp_dir"
    return 1
  fi

  token="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
  if ! remote_bash "set -euo pipefail
mkdir -p $(shell_quote "$REMOTE_WORKFLOW_RUN_ROOT")
if ! mkdir $(shell_quote "$root"); then
  echo $(shell_quote "Run already exists and will not be overwritten: $root") >&2
  exit 73
fi
mkdir -p $(shell_quote "$root/source") $(shell_quote "$root/.state") $(shell_quote "$root/optuna/workers") $(shell_quote "$root/logs") $(shell_quote "$root/outputs")
printf '%s\n' $(shell_quote "$token") > $(shell_quote "$root/.state/preparing")"; then
    rm -rf -- "$tmp_dir"
    return 1
  fi

  if ! rsync -az "$snapshot/" "$TRANSFER_HOST:$root/source/"; then
    cleanup_failed_run "$root" "$token"
    rm -rf -- "$tmp_dir"
    return 1
  fi
  if ! rsync -az "$rendered" "$TRANSFER_HOST:$root/config.yaml"; then
    cleanup_failed_run "$root" "$token"
    rm -rf -- "$tmp_dir"
    return 1
  fi
  if ! rsync -az "$file_list" "$TRANSFER_HOST:$root/.state/source-files.bin"; then
    cleanup_failed_run "$root" "$token"
    rm -rf -- "$tmp_dir"
    return 1
  fi

  if ! remote_bash "set -euo pipefail
test \"\$(cat $(shell_quote "$root/.state/preparing"))\" = $(shell_quote "$token")
cd $(shell_quote "$root/source")
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(shell_quote "$root/source") \\
  $(shell_quote "$REMOTE_TOOLING_ENV/bin/python") \\
  -m scripts.workflow.run_config verify \\
  --config $(shell_quote "$root/config.yaml") \\
  --source-root $(shell_quote "$root/source") \\
  --file-list $(shell_quote "$root/.state/source-files.bin")
chmod -R a-w $(shell_quote "$root/source")
chmod 0444 $(shell_quote "$root/config.yaml") $(shell_quote "$root/.state/source-files.bin")
printf '%s\n' $(shell_quote "$run_id") > $(shell_quote "$root/.state/run-id")
rm -f $(shell_quote "$root/.state/preparing")"; then
    cleanup_failed_run "$root" "$token"
    rm -rf -- "$tmp_dir"
    return 1
  fi
  rm -rf -- "$tmp_dir"

  echo "Prepared immutable MN5 workflow run"
  echo "  run_id: $run_id"
  echo "  run_root: $root"
  echo "  tooling_key: $TOOLING_KEY"
  echo "  container: $resolved_sif"
}

dry_run() {
  local run_id="$1"
  shift
  validate_run_id "$run_id"
  ensure_tooling
  remote_bash "$(snakemake_command "$run_id" --dry-run "$@")"
}

start_run() {
  local run_id="$1"
  shift
  local root coordinator submit launch result allocation_output
  local account partition qos nodes tasks cpus_per_task gres runtime
  validate_run_id "$run_id"
  ensure_tooling
  root="$(run_root "$run_id")"
  coordinator="$root/source/scripts/workflow/snakemake_coordinator.sbatch"
  allocation_output="$(remote_bash "
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(shell_quote "$root/source") \\
  $(shell_quote "$REMOTE_TOOLING_ENV/bin/python") \\
  -m scripts.workflow.run_config allocation \\
  --config $(shell_quote "$root/config.yaml")")"
  IFS=$'\t' read -r account partition qos nodes tasks cpus_per_task gres runtime <<< "$allocation_output"
  if [ -z "$runtime" ]; then
    echo "Could not read the immutable MN5 allocation from $root/config.yaml" >&2
    return 1
  fi
  submit="sbatch --parsable \\
  --chdir=$(shell_quote "$root") \\
  --output=$(shell_quote "$root/logs/snakemake-%j.out") \\
  --error=$(shell_quote "$root/logs/snakemake-%j.err") \\
  --open-mode=append \\
  --account=$(shell_quote "$account") \\
  --partition=$(shell_quote "$partition") \\
  --qos=$(shell_quote "$qos") \\
  --nodes=$(shell_quote "$nodes") \\
  --ntasks=$(shell_quote "$tasks") \\
  --ntasks-per-node=$(shell_quote "$tasks") \\
  --cpus-per-task=$(shell_quote "$cpus_per_task") \\
  --gres=$(shell_quote "$gres") \\
  --time=$(shell_quote "$runtime") \\
  $(shell_quote "$coordinator") \\
  $(shell_quote "$root") \\
  $(shell_quote "$REMOTE_SNAKEMAKE_BIN") \\
  $(shell_quote "$tasks") \\
  $(shell_quote "$cpus_per_task")"
  for arg in "$@"; do
    submit+=" $(shell_quote "$arg")"
  done

  launch="set -euo pipefail
test -f $(shell_quote "$root/config.yaml")
test -x $(shell_quote "$coordinator")
exec 9>$(shell_quote "$root/.state/coordinator.launch.lock")
flock 9
if test -f $(shell_quote "$root/.state/workflow.done"); then
  printf 'complete'
  exit 0
fi
if test -s $(shell_quote "$root/.state/coordinator.job_id"); then
  previous=\$(cat $(shell_quote "$root/.state/coordinator.job_id"))
  state=\$(squeue -h -j \"\$previous\" -o '%T' 2>/dev/null | head -n 1 || true)
  case \"\$state\" in
    PENDING|RUNNING|CONFIGURING|COMPLETING)
      printf 'active:%s:%s' \"\$previous\" \"\$state\"
      exit 0
      ;;
  esac
fi
job_id=\$($submit)
job_id=\${job_id%%;*}
printf '%s\n' \"\$job_id\" > $(shell_quote "$root/.state/coordinator.job_id")
mkdir -p $(shell_quote "$REMOTE_WORKFLOW_RUN_ROOT/by-slurm")
ln -s $(shell_quote "$root") $(shell_quote "$REMOTE_WORKFLOW_RUN_ROOT/by-slurm")/\"\${job_id}_0\"
printf 'submitted:%s' \"\$job_id\""
  result="$(remote_bash "$launch")"
  case "$result" in
    submitted:*)
      echo "Submitted Snakemake coordinator job ${result#submitted:}"
      ;;
    active:*)
      result="${result#active:}"
      echo "Snakemake coordinator is already active: ${result%%:*} (${result#*:})"
      ;;
    complete)
      echo "Workflow is already complete: $run_id"
      ;;
    *)
      echo "Unexpected coordinator submission response: $result" >&2
      return 1
      ;;
  esac
  echo "Run root: $root"
}

show_status() {
  local run_id="$1"
  local root command
  validate_run_id "$run_id"
  root="$(run_root "$run_id")"
  command="set -euo pipefail
job_id=''
if test -f $(shell_quote "$root/.state/coordinator.job_id"); then job_id=\$(cat $(shell_quote "$root/.state/coordinator.job_id")); fi
if test -n \"\$job_id\"; then
  state=\$(squeue -h -j \"\$job_id\" -o '%T' 2>/dev/null | head -n 1 || true)
  if test -n \"\$state\"; then
    echo \"coordinator=\$state job_id=\$job_id execution_id=\${job_id}_0\"
  else
    sacct -n -X -j \"\$job_id\" --format=JobIDRaw,State,ExitCode 2>/dev/null | head -n 1 || true
  fi
else
  echo 'coordinator=not-submitted'
fi
test -f $(shell_quote "$root/.state/workflow.done") && echo workflow=complete || echo workflow=incomplete
if test -n \"\$job_id\"; then
  tail -n 40 $(shell_quote "$root/logs")/snakemake-\"\$job_id\".out 2>/dev/null || true
  tail -n 40 $(shell_quote "$root/logs")/snakemake-\"\$job_id\".err 2>/dev/null || true
fi"
  remote_bash "$command"
}

unlock_run() {
  local run_id="$1"
  local root
  validate_run_id "$run_id"
  root="$(run_root "$run_id")"
  remote_bash "set -euo pipefail
if test -s $(shell_quote "$root/.state/coordinator.job_id"); then
  job_id=\$(cat $(shell_quote "$root/.state/coordinator.job_id"))
  state=\$(squeue -h -j \"\$job_id\" -o '%T' 2>/dev/null | head -n 1 || true)
  if test -n \"\$state\"; then echo \"Coordinator job \$job_id is still \$state\" >&2; exit 1; fi
fi
$(snakemake_command "$run_id" --unlock)"
}

command="${1:-}"
case "$command" in
  tooling)
    shift
    [ "$#" -eq 0 ] || { usage >&2; exit 2; }
    ensure_tooling
    ;;
  prepare)
    shift
    run_id=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --config)
          [ "$#" -ge 2 ] || { usage >&2; exit 2; }
          TEMPLATE="$2"
          shift 2
          ;;
        *)
          [ -z "$run_id" ] || { usage >&2; exit 2; }
          run_id="$1"
          shift
          ;;
      esac
    done
    [ -f "$TEMPLATE" ] || { echo "Workflow template not found: $TEMPLATE" >&2; exit 2; }
    prepare_run "$run_id"
    ;;
  dry-run)
    shift
    [ "$#" -ge 1 ] || { usage >&2; exit 2; }
    run_id="$1"
    shift
    dry_run "$run_id" "$@"
    ;;
  run)
    shift
    [ "$#" -ge 1 ] || { usage >&2; exit 2; }
    run_id="$1"
    shift
    start_run "$run_id" "$@"
    ;;
  status)
    shift
    [ "$#" -eq 1 ] || { usage >&2; exit 2; }
    show_status "$1"
    ;;
  unlock)
    shift
    [ "$#" -eq 1 ] || { usage >&2; exit 2; }
    unlock_run "$1"
    ;;
  path)
    shift
    [ "$#" -eq 1 ] || { usage >&2; exit 2; }
    validate_run_id "$1"
    run_root "$1"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash

# Validate the complete maintained shell surface with the repository's pinned
# formatter. The Bash/Zsh source helper is syntax-checked but intentionally not
# passed to shfmt or ShellCheck because it contains a zsh parameter expansion.

set -euo pipefail

REQUIRED_SHFMT_VERSION="v3.13.1"

MAINTAINED_SHELL_FILES=(
  containers/runtime/build_sif.sbatch
  containers/runtime/build_sif_on_mn5.sh
  containers/runtime/fetch_runtime_deps_on_wifi.sh
  scripts/check_shell_quality.sh
  scripts/cluster/diagnose_mn5_env.sh
  scripts/sync_codex_skills.sh
  scripts/workflow/launch_parallel_workers.sh
  scripts/workflow/mn5_container_exec.sh
  scripts/workflow/mn5_workflow.sh
  scripts/workflow/remote_run.sh
  scripts/workflow/remote_tooling.sh
  scripts/workflow/run_parallel_worker.sh
  scripts/workflow/snakemake_coordinator.sbatch
  spells.sh
)

FORMATTED_BASH_FILES=(
  containers/runtime/build_sif.sbatch
  containers/runtime/build_sif_on_mn5.sh
  containers/runtime/fetch_runtime_deps_on_wifi.sh
  scripts/check_shell_quality.sh
  scripts/cluster/diagnose_mn5_env.sh
  scripts/sync_codex_skills.sh
  scripts/workflow/launch_parallel_workers.sh
  scripts/workflow/mn5_container_exec.sh
  scripts/workflow/mn5_workflow.sh
  scripts/workflow/remote_run.sh
  scripts/workflow/remote_tooling.sh
  scripts/workflow/run_parallel_worker.sh
  scripts/workflow/snakemake_coordinator.sbatch
)

# Resolve paths from the repository root regardless of the caller's directory.
enter_repository() {
  local script_dir

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/.."
}

# Require the same formatter release used to establish the checked-in layout.
verify_shfmt_version() {
  local actual_version

  if ! command -v shfmt >/dev/null 2>&1; then
    echo "shfmt $REQUIRED_SHFMT_VERSION is required" >&2
    return 1
  fi

  actual_version="$(shfmt --version)"
  if [ "$actual_version" != "$REQUIRED_SHFMT_VERSION" ]; then
    echo "Expected shfmt $REQUIRED_SHFMT_VERSION, found $actual_version" >&2
    return 1
  fi
}

# Parse every maintained script with Bash, including the cross-shell helper.
check_bash_syntax() {
  local file

  for file in "${MAINTAINED_SHELL_FILES[@]}"; do
    bash -n "$file"
  done
}

# EditorConfig supplies the deterministic formatting policy.
check_formatting() {
  shfmt -d "${FORMATTED_BASH_FILES[@]}"
}

# ShellCheck is optional locally but runs across every Bash-only maintained file
# whenever the executable is available.
check_shellcheck() {
  if ! command -v shellcheck >/dev/null 2>&1; then
    echo "ShellCheck unavailable; skipped static shell analysis" >&2
    return 0
  fi

  shellcheck "${FORMATTED_BASH_FILES[@]}"
}

main() {
  enter_repository
  verify_shfmt_version
  check_bash_syntax
  check_formatting
  check_shellcheck
  echo "Shell quality checks passed for ${#MAINTAINED_SHELL_FILES[@]} files"
}

main "$@"

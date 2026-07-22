#!/bin/bash
set -euo pipefail

# Print immutable context before probing optional cluster services.
print_environment() {
  echo "=== MN5 Environment Diagnostics ==="
  echo "Date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Host:    $(hostname)"
  echo "User:    $(whoami)"
  echo "SLURM:   ${SLURM_CLUSTER_NAME:-<not set>}"
  echo "Node:    ${SLURMD_NODENAME:-<not in job>}"
  echo "Account: ${SLURM_JOB_ACCOUNT:-<not in job>}"
  echo "QoS:     ${SLURM_JOB_QOS:-<not in job>}"
}

# Load BSC profile scripts only when the module shell function is unavailable.
diagnose_module_system() {
  local profile_script

  echo "--- Module System ---"
  if command -v module >/dev/null 2>&1; then
    echo "  module command: found ($(command -v module))"
    return 0
  fi

  echo "  module command: NOT FOUND (likely a transfer node)"
  echo "  Trying to source BSC profiles..."
  for profile_script in /etc/profile.d/00-bsc.sh /etc/profile.d/01-module.sh; do
    if [ -r "$profile_script" ]; then
      set +u
      # shellcheck disable=SC1090
      source "$profile_script" 2>/dev/null
      set -u
      echo "    sourced $profile_script"
    fi
  done

  if command -v module >/dev/null 2>&1; then
    echo "  module now available after sourcing profiles"
  else
    echo "  module still unavailable"
  fi
}

# Report whether each module documented for MN5 is visible from this endpoint.
check_documented_modules() {
  local module_name
  local modules=(EB/apps EB/install gcc/12.3.0 CUDA/12.1.1 cmake/4.1.2-gcc)

  echo "--- Checking documented modules ---"
  for module_name in "${modules[@]}"; do
    if ! command -v module >/dev/null 2>&1; then
      echo "  SKIPPED  $module_name (no module command)"
      continue
    fi

    if module avail "$module_name" 2>&1 | grep -q "$module_name"; then
      echo "  FOUND    $module_name"
    else
      echo "  MISSING  $module_name"
    fi
  done
}

# Show the user's account/QoS association when this Slurm controller exposes it.
show_associations() {
  local associations

  echo "--- QoS / Account Associations ---"
  associations="$(
    sacctmgr show assoc where user="$USER" \
      format=account,user,qos,partition -P -n 2>/dev/null | tr -s ' '
  )"
  if [ -z "$associations" ]; then
    echo "  No associations found (transfer nodes may not see compute QoS)."
    echo "  Run this script from a compute login node (ssh mn5) for full results."
  else
    echo "  Account  User        QOS                                      Partition"
    echo "  $associations"
  fi
}

# Explain whether this host is suitable for scheduler operations.
show_node_type() {
  local hostname

  echo "--- Node Type Check ---"
  hostname="$(hostname)"
  case "$hostname" in
    transfer*)
      echo "  DETECTED: transfer node ($hostname)"
      echo "  This node is for data transfer (rsync) only."
      echo "  Job submission will fail or route to wrong partition."
      echo "  Use: ssh mn5 (glogin1.bsc.es) for sbatch/squeue/sacct."
      ;;
    glogin* | alogin*)
      echo "  DETECTED: compute login node ($hostname)"
      echo "  OK for job submission and monitoring."
      ;;
    *)
      echo "  UNKNOWN node type: $hostname"
      ;;
  esac
}

main() {
  print_environment
  echo
  diagnose_module_system
  echo
  check_documented_modules
  echo
  show_associations
  echo
  show_node_type
  echo
  echo "=== Done ==="
}

main "$@"

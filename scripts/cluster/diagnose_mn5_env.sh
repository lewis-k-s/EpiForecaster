#!/bin/bash
set -euo pipefail

echo "=== MN5 Environment Diagnostics ==="
echo "Date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host:    $(hostname)"
echo "User:    $(whoami)"
echo "SLURM:   ${SLURM_CLUSTER_NAME:-<not set>}"
echo "Node:    ${SLURMD_NODENAME:-<not in job>}"
echo "Account: ${SLURM_JOB_ACCOUNT:-<not in job>}"
echo "QoS:     ${SLURM_JOB_QOS:-<not in job>}"
echo ""

echo "--- Module System ---"
if command -v module >/dev/null 2>&1; then
  echo "  module command: found ($(which module))"
else
  echo "  module command: NOT FOUND (likely a transfer node)"
  echo "  Trying to source BSC profiles..."
  for f in /etc/profile.d/00-bsc.sh /etc/profile.d/01-module.sh; do
    if [ -r "$f" ]; then
      # shellcheck disable=SC1090
      set +u; source "$f" 2>/dev/null; set -u
      echo "    sourced $f"
    fi
  done
  if command -v module >/dev/null 2>&1; then
    echo "  module now available after sourcing profiles"
  else
    echo "  module still unavailable"
  fi
fi

echo ""
echo "--- Checking documented modules ---"
for mod in EB/apps EB/install gcc/12.3.0 CUDA/12.1.1 cmake/4.1.2-gcc; do
  if command -v module >/dev/null 2>&1; then
    if module avail "$mod" 2>&1 | grep -q "$mod"; then
      echo "  FOUND    $mod"
    else
      echo "  MISSING  $mod"
    fi
  else
    echo "  SKIPPED  $mod (no module command)"
  fi
done

echo ""
echo "--- QoS / Account Associations ---"
assoc=$(sacctmgr show assoc where user="$USER" format=account,user,qos,partition -P -n 2>/dev/null | tr -s ' ')
if [ -z "$assoc" ]; then
  echo "  No associations found (transfer nodes may not see compute QoS)."
  echo "  Run this script from a compute login node (ssh mn5) for full results."
else
  echo "  Account  User        QOS                                      Partition"
  echo "  $assoc"
fi

echo ""
echo "--- Node Type Check ---"
hn=$(hostname)
case "$hn" in
  transfer*)
    echo "  DETECTED: transfer node ($hn)"
    echo "  This node is for data transfer (rsync) only."
    echo "  Job submission will fail or route to wrong partition."
    echo "  Use: ssh mn5 (glogin1.bsc.es) for sbatch/squeue/sacct."
    ;;
  glogin*|alogin*)
    echo "  DETECTED: compute login node ($hn)"
    echo "  OK for job submission and monitoring."
    ;;
  *)
    echo "  UNKNOWN node type: $hn"
    ;;
esac

echo ""
echo "=== Done ==="

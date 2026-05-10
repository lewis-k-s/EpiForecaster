#!/bin/bash

# Shared MareNostrum 5 module initialization for batch scripts.
#
# Environment knobs:
#   MN5_LOAD_CUDA=0   Skip CUDA module load
#   MN5_LOAD_CMAKE=0  Skip CMake module load
#   MN5_SETUP_VERBOSE=1  Print loaded toolchain summary

set -euo pipefail

_mn5_fail() {
  echo "$1" >&2
  return 1
}

if ! command -v module >/dev/null 2>&1; then
  # Slurm batch shells on MN5 do not always source /etc/profile.d.
  # Temporarily relax nounset because the site profile scripts reference
  # variables before defining them.
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

if ! command -v module >/dev/null 2>&1; then
  echo "HPC modules not found!"
  exit 1
fi

command -v module >/dev/null 2>&1 || _mn5_fail "Environment modules command not found."

module load EB/apps || _mn5_fail "EB/apps module not found."
module load GCC/12.3.0 || _mn5_fail "GCC module not found."

if [ "${MN5_LOAD_CUDA:-1}" = "1" ]; then
  module load CUDA/12.1.1 || _mn5_fail "CUDA module not found."
fi

if [ "${MN5_LOAD_CMAKE:-1}" = "1" ]; then
  module load cmake/4.1.2-gcc || _mn5_fail "CMake module not found."
fi

export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

if [ "${MN5_SETUP_VERBOSE:-0}" = "1" ]; then
  echo "MN5 modules initialized: CC=${CC} CXX=${CXX} CUDA=${MN5_LOAD_CUDA:-1} CMAKE=${MN5_LOAD_CMAKE:-1}"
fi

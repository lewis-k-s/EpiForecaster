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
  if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
fi

command -v module >/dev/null 2>&1 || _mn5_fail "Environment modules command not found."

module load EB/apps EB/install || _mn5_fail "Core EB modules not found."
module load gcc/12.3.0 || _mn5_fail "GCC module not found."

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

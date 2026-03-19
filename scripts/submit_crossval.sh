#!/bin/bash

set -euo pipefail

derive_array_spec_from_seeds() {
  local cv_seeds="$1"
  local -a seed_list=()

  read -r -a seed_list <<<"$cv_seeds"
  if [ "${#seed_list[@]}" -eq 0 ]; then
    echo "Error: CV_SEEDS must contain at least one seed" >&2
    return 1
  fi

  printf '0-%d\n' "$(( ${#seed_list[@]} - 1 ))"
}

parse_explicit_array_arg() {
  local expect_value=0
  local arg

  for arg in "$@"; do
    if [ "$expect_value" -eq 1 ]; then
      printf '%s\n' "$arg"
      return 0
    fi

    case "$arg" in
      --array)
        expect_value=1
        ;;
      --array=*)
        printf '%s\n' "${arg#--array=}"
        return 0
        ;;
    esac
  done

  if [ "$expect_value" -eq 1 ]; then
    echo "Error: --array provided without a value" >&2
    return 2
  fi

  return 1
}

main() {
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CAMPAIGN_ID="${CAMPAIGN_ID:-crossval_$(date +%s)}"
CV_SEEDS="${CV_SEEDS:-42 43 44 45 46}"
CV_ARRAY_SPEC="${CV_ARRAY_SPEC:-}"

export PROJECT_ROOT
export CAMPAIGN_ID

if explicit_array_spec="$(parse_explicit_array_arg "$@")"; then
  array_spec="$explicit_array_spec"
  array_args=()
elif [ -n "$CV_ARRAY_SPEC" ]; then
  array_spec="$CV_ARRAY_SPEC"
  array_args=(--array "$array_spec")
else
  array_spec="$(derive_array_spec_from_seeds "$CV_SEEDS")"
  array_args=(--array "$array_spec")
fi

ARRAY_JOB_ID=$(sbatch --parsable "${array_args[@]}" "$@" scripts/run_crossval.sbatch)
AGG_JOB_ID=$(
  sbatch \
    --parsable \
    --dependency=afterok:${ARRAY_JOB_ID} \
    scripts/run_crossval_aggregate.sbatch
)

echo "Submitted cross-val array job: ${ARRAY_JOB_ID}"
echo "Submitted aggregate job: ${AGG_JOB_ID}"
echo "Campaign ID: ${CAMPAIGN_ID}"
echo "Array spec: ${array_spec}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi

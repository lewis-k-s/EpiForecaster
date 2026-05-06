#!/bin/bash

set -euo pipefail

EPIFORECASTER_PATH="${EPIFORECASTER_PATH:-/home/bsc/bsc008913/EpiForecaster}"

usage() {
  cat <<'EOF'
Usage:
  mn5_dispatch.sh submit <mode> [sbatch-args...]
  mn5_dispatch.sh status <job-id>
  mn5_dispatch.sh logs <job-id> [task-id]
  mn5_dispatch.sh tail <job-id> [task-id] [lines]

Submit modes:
  single            scripts/cluster/train_single_gpu.sbatch
  optuna            scripts/cluster/optuna_epiforecaster.sbatch
  missing-permit    scripts/cluster/optuna_epiforecaster_missing_permit.sbatch
  context-length    scripts/cluster/optuna_epiforecaster_context_length.sbatch
  ablation          scripts/cluster/optuna_ablation.sbatch
  foldval           scripts/cluster/run_fold_validation.sbatch
  crossval          scripts/cluster/submit_crossval.sh
  synth-pretrain    scripts/cluster/submit_synth_pretrain.sh
  pretrain-finetune scripts/cluster/submit_pretrain_then_finetune.sh
  pretrain-real-eval scripts/cluster/eval_pretrain_on_real.sbatch

Examples:
  ssh mn5 'DRY_RUN=1 /home/bsc/bsc008913/EpiForecaster/scripts/cluster/mn5_dispatch.sh submit single --time=00:10:00'
  ssh mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/mn5_dispatch.sh tail 39875209'
  ssh mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/mn5_dispatch.sh tail 39875209 0 200'
EOF
}

repo_cd() {
  cd "$EPIFORECASTER_PATH"
  export PROJECT_ROOT="${PROJECT_ROOT:-$EPIFORECASTER_PATH}"
}

submit_script_for_mode() {
  local mode="$1"

  case "$mode" in
    single) printf '%s\n' "scripts/cluster/train_single_gpu.sbatch" ;;
    optuna) printf '%s\n' "scripts/cluster/optuna_epiforecaster.sbatch" ;;
    missing-permit)
      printf '%s\n' "scripts/cluster/optuna_epiforecaster_missing_permit.sbatch"
      ;;
    context-length)
      printf '%s\n' "scripts/cluster/optuna_epiforecaster_context_length.sbatch"
      ;;
    ablation) printf '%s\n' "scripts/cluster/optuna_ablation.sbatch" ;;
    foldval) printf '%s\n' "scripts/cluster/run_fold_validation.sbatch" ;;
    crossval) printf '%s\n' "scripts/cluster/submit_crossval.sh" ;;
    synth-pretrain) printf '%s\n' "scripts/cluster/submit_synth_pretrain.sh" ;;
    pretrain-finetune)
      printf '%s\n' "scripts/cluster/submit_pretrain_then_finetune.sh"
      ;;
    pretrain-real-eval)
      printf '%s\n' "scripts/cluster/eval_pretrain_on_real.sbatch"
      ;;
    *)
      echo "Unknown submit mode: $mode" >&2
      return 1
      ;;
  esac
}

submit_job() {
  local mode="${1:-}"
  local script_path

  if [ -z "$mode" ]; then
    usage >&2
    return 2
  fi
  shift

  repo_cd
  script_path="$(submit_script_for_mode "$mode")"

  case "$mode" in
    crossval | synth-pretrain | pretrain-finetune)
      "$script_path" "$@"
      ;;
    *)
      sbatch --parsable "$@" "$script_path"
      ;;
  esac
}

show_status() {
  local job_id="${1:-}"

  if [ -z "$job_id" ]; then
    usage >&2
    return 2
  fi

  repo_cd
  squeue --job "$job_id" --format="%.18i %.9P %.32j %.8u %.2t %.10M %.10l %.6D %R" || true
  sacct --jobs "$job_id" --format=JobID,JobName%32,State,ExitCode,Elapsed,Timelimit,NodeList --parsable2 || true
}

resolve_log_files() {
  local job_id="$1"
  local task_id="${2:-}"
  local pattern
  local -a patterns=()

  if [ -n "$task_id" ]; then
    patterns+=("logs/slurm-*-${job_id}_${task_id}.out")
    patterns+=("logs/slurm-*-${job_id}_${task_id}.err")
  else
    patterns+=("logs/slurm-*-${job_id}.out")
    patterns+=("logs/slurm-*-${job_id}.err")
    patterns+=("logs/slurm-*-${job_id}_*.out")
    patterns+=("logs/slurm-*-${job_id}_*.err")
  fi

  for pattern in "${patterns[@]}"; do
    compgen -G "$pattern" || true
  done | sort -u
}

show_logs() {
  local job_id="${1:-}"
  local task_id="${2:-}"
  local -a files=()
  local file

  if [ -z "$job_id" ]; then
    usage >&2
    return 2
  fi

  repo_cd
  while IFS= read -r file; do
    files+=("$file")
  done < <(resolve_log_files "$job_id" "$task_id")

  if [ "${#files[@]}" -eq 0 ]; then
    echo "No Slurm logs found for job ${job_id}${task_id:+ task ${task_id}} under $EPIFORECASTER_PATH/logs" >&2
    return 1
  fi

  printf '%s\n' "${files[@]}"
}

tail_logs() {
  local job_id="${1:-}"
  local maybe_task="${2:-}"
  local maybe_lines="${3:-}"
  local task_id=""
  local lines="120"
  local file
  local -a files=()

  if [ -z "$job_id" ]; then
    usage >&2
    return 2
  fi

  if [ -n "$maybe_task" ]; then
    task_id="$maybe_task"
  fi
  if [ -n "$maybe_lines" ]; then
    lines="$maybe_lines"
  elif [ -n "$maybe_task" ] && [[ "$maybe_task" =~ ^[0-9]+$ ]]; then
    if ! compgen -G "$EPIFORECASTER_PATH/logs/slurm-*-${job_id}_${maybe_task}.out" >/dev/null \
      && ! compgen -G "$EPIFORECASTER_PATH/logs/slurm-*-${job_id}_${maybe_task}.err" >/dev/null; then
      task_id=""
      lines="$maybe_task"
    fi
  fi

  repo_cd
  while IFS= read -r file; do
    files+=("$file")
  done < <(resolve_log_files "$job_id" "$task_id")

  if [ "${#files[@]}" -eq 0 ]; then
    echo "No Slurm logs found for job ${job_id}${task_id:+ task ${task_id}} under $EPIFORECASTER_PATH/logs" >&2
    return 1
  fi

  tail -n "$lines" -f "${files[@]}"
}

main() {
  local command="${1:-}"

  case "$command" in
    submit)
      shift
      submit_job "$@"
      ;;
    status)
      shift
      show_status "$@"
      ;;
    logs)
      shift
      show_logs "$@"
      ;;
    tail)
      shift
      tail_logs "$@"
      ;;
    -h | --help | help)
      usage
      ;;
    *)
      usage >&2
      return 2
      ;;
  esac
}

main "$@"

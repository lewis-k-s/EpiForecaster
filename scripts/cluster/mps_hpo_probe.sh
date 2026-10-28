#!/bin/bash

set -euo pipefail

# Measure how K concurrent HPO trials share one MN5 GPU, with and without CUDA
# MPS. Each condition runs K real-data `hpsearch_worker` trials on one GPU
# inside one Slurm step of an existing allocation, samples that GPU once per
# second, and writes everything below PROBE_ROOT/results/<condition>/.
#
# The probe never edits a deployed bundle or run root. It reads an immutable
# template run (source bundle and materialized DVC inputs) and writes only
# below PROBE_ROOT and node-local scratch.
#
#   mps_hpo_probe.sh sweep     ...   login node: one Slurm step per condition
#   mps_hpo_probe.sh condition ...   inside a step: stage, run K workers, sample

MODE_COMMAND=""
ALLOCATION=""
PROBE_ROOT=""
TEMPLATE_RUN=""
INPUTS_RUN=""
SIF=""
DVC_CACHE=""
GPU_UUID=""
CONDITIONS="plain:1 mps:1 plain:2 plain:4 mps:2 mps:4"
EPOCHS="5"
CPUS="20"
MPS_THREAD_PCT=""
STEP_GPUS="1"
STEP_CPUS=""
CPU_LIST=""
COMPILE_MODE=""
CONDITION_TIMEOUT="1200"
PROFILE="0"
DETERMINISTIC="0"
EXACT_KERNELS="0"
STUB=""
TRIALS="1"
EXTRA_OVERRIDES=()
REGION_EMBEDDINGS_MD5="5205439ed1bd8a079aa57aed37507654"
CONDITION=""
STUDY_PREFIX="mpsprobe"
WORKER_PIDS=()
SAMPLER_PID=""
PCIE_SAMPLER_PID=""
MPS_PIPE_DIR=""

# Print the probe interface.
usage() {
  cat <<EOF
Usage:
  $(basename "$0") sweep --allocation JOB --probe-root PATH --template-run PATH \\
    --sif PATH --dvc-cache PATH --gpu-uuid GPU-UUID [--inputs-run PATH] \\
    [--conditions "plain:1 mps:1 plain:2 mps:2 plain:4 mps:4"] \\
    [--epochs 5] [--cpus 20] [--mps-thread-pct N] [--step-gpus 1] \\
    [--cpu-list 0-19] [--compile-mode default] [--condition-timeout 1200]
  $(basename "$0") condition ... --condition MODE:K   (run inside a Slurm step)

MODE is plain (default time slicing) or mps (CUDA MPS daemon). K is the number
of concurrent trials on the one GPU. --cpus is the CPU budget for the whole GPU
and defaults to the MN5 contract of 20 CPUs per GPU. --step-gpus is the GPU
count of each overlapping step (all allocation GPUs stay visible; --gpu-uuid
selects one). --cpu-list pins the workers so parallel sweeps on one node use
disjoint cores; overlapping steps get the allocation's first cores, so set
--step-cpus to the allocation CPU count when the list names other cores. --compile-mode overrides training.compile_mode.
--condition-timeout stops a stalled condition after that many seconds.
--profile runs each worker under cProfile (worker-N.prof) and enables the
final validation/test evaluation so the granular writer is measured too.
--override KEY=VALUE (repeatable) adds a training config override.
--trials N runs N sequential trials in each worker process (default 1).
--deterministic sets torch.use_deterministic_algorithms(True) and the cuBLAS
workspace config before the worker starts (about 70x slower before 99e943c).
--exact-kernels gives every inductor Triton kernel one config, so compiled
training is bitwise reproducible run to run (about 3-5% slower).
--stub FILE runs FILE (bound read-only) instead of the worker module; FILE
receives the worker arguments and must start the worker itself.
EOF
}

# Parse the shared option set for both subcommands.
parse_arguments() {
  MODE_COMMAND="${1:-}"
  shift || true
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --allocation) ALLOCATION="$2"; shift 2 ;;
      --probe-root) PROBE_ROOT="$2"; shift 2 ;;
      --template-run) TEMPLATE_RUN="$2"; shift 2 ;;
      --inputs-run) INPUTS_RUN="$2"; shift 2 ;;
      --sif) SIF="$2"; shift 2 ;;
      --dvc-cache) DVC_CACHE="$2"; shift 2 ;;
      --gpu-uuid) GPU_UUID="$2"; shift 2 ;;
      --conditions) CONDITIONS="$2"; shift 2 ;;
      --epochs) EPOCHS="$2"; shift 2 ;;
      --cpus) CPUS="$2"; shift 2 ;;
      --mps-thread-pct) MPS_THREAD_PCT="$2"; shift 2 ;;
      --step-gpus) STEP_GPUS="$2"; shift 2 ;;
      --step-cpus) STEP_CPUS="$2"; shift 2 ;;
      --cpu-list) CPU_LIST="$2"; shift 2 ;;
      --compile-mode) COMPILE_MODE="$2"; shift 2 ;;
      --condition-timeout) CONDITION_TIMEOUT="$2"; shift 2 ;;
      --profile) PROFILE="1"; shift ;;
      --deterministic) DETERMINISTIC="1"; shift ;;
      --exact-kernels) EXACT_KERNELS="1"; shift ;;
      --stub) STUB="$2"; shift 2 ;;
      --trials) TRIALS="$2"; shift 2 ;;
      --override) EXTRA_OVERRIDES+=("$2"); shift 2 ;;
      --condition) CONDITION="$2"; shift 2 ;;
      *) usage >&2; return 2 ;;
    esac
  done
}

# Reject missing or malformed inputs before any scheduler or GPU work.
validate_arguments() {
  if [ -z "$PROBE_ROOT" ] || [ -z "$TEMPLATE_RUN" ] || [ -z "$SIF" ] \
    || [ -z "$DVC_CACHE" ] || [ -z "$GPU_UUID" ]; then
    usage >&2
    return 2
  fi
  if [[ ! "$GPU_UUID" =~ ^GPU-[0-9a-f-]+$ ]]; then
    echo "--gpu-uuid must be a GPU UUID, not an index: $GPU_UUID" >&2
    return 2
  fi
  if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || [ "$EPOCHS" -lt 4 ]; then
    echo "--epochs must be at least 4 (epoch 0 compiles; 1..N-2 are timed)" >&2
    return 2
  fi
  INPUTS_RUN="${INPUTS_RUN:-$TEMPLATE_RUN}"
  test -f "$SIF"
  test -d "$TEMPLATE_RUN/source"
  test -d "$INPUTS_RUN/.state/dvc-workspace"
  check_input_pointers
}

# A separate inputs run must materialize the same DVC objects the source
# bundle declares; compare the pointer hashes rather than trusting names.
check_input_pointers() {
  local pointer

  [ "$INPUTS_RUN" = "$TEMPLATE_RUN" ] && return 0
  for pointer in data/files/mobility.zarr.dvc data/processed/real_with_id.zarr.dvc \
    outputs/region_embeddings/region_embeddings.pt.dvc; do
    if ! cmp -s <(grep -m1 md5 "$TEMPLATE_RUN/source/$pointer") \
      <(grep -m1 md5 "$INPUTS_RUN/source/$pointer"); then
      echo "Input pointer differs between template and inputs run: $pointer" >&2
      return 1
    fi
  done
}

# Run one Slurm step per condition so Slurm accounts each one separately.
run_sweep() {
  local condition mode workers override
  local -a passthrough=()

  : "${ALLOCATION:?--allocation is required for sweep}"
  for override in "${EXTRA_OVERRIDES[@]}"; do
    passthrough+=(--override "$override")
  done
  mkdir -p "$PROBE_ROOT/results" "$PROBE_ROOT/logs"
  for condition in $CONDITIONS; do
    mode="${condition%%:*}"
    workers="${condition##*:}"
    if [ -e "$PROBE_ROOT/results/$mode-k$workers" ]; then
      echo "Skipping $condition: result directory exists" >&2
      continue
    fi
    echo "=== $(date -u +%H:%M:%SZ) condition $condition"
    # All allocation GPUs stay visible to the step; one is chosen by UUID.
    srun \
      --jobid="$ALLOCATION" --overlap --nodes=1 --ntasks=1 \
      --cpus-per-task="${STEP_CPUS:-$CPUS}" --gres="gpu:$STEP_GPUS" \
      --job-name="mps-probe-$mode-k$workers" \
      "$PROBE_ROOT/mps_hpo_probe.sh" condition \
      --probe-root "$PROBE_ROOT" --template-run "$TEMPLATE_RUN" \
      --inputs-run "$INPUTS_RUN" --sif "$SIF" \
      --dvc-cache "$DVC_CACHE" --gpu-uuid "$GPU_UUID" --epochs "$EPOCHS" \
      --cpus "$CPUS" --condition "$condition" \
      --condition-timeout "$CONDITION_TIMEOUT" --trials "$TRIALS" \
      ${MPS_THREAD_PCT:+--mps-thread-pct "$MPS_THREAD_PCT"} \
      ${CPU_LIST:+--cpu-list "$CPU_LIST"} \
      ${COMPILE_MODE:+--compile-mode "$COMPILE_MODE"} \
      $([ "$PROFILE" = "1" ] && echo --profile) \
      $([ "$DETERMINISTIC" = "1" ] && echo --deterministic) \
      $([ "$EXACT_KERNELS" = "1" ] && echo --exact-kernels) \
      ${STUB:+--stub "$STUB"} \
      "${passthrough[@]}" \
      >"$PROBE_ROOT/logs/$mode-k$workers.step.log" 2>&1 \
      || echo "condition $condition exited with status $?" >&2
  done
}

# Load Singularity the same way mn5_container_exec.sh does.
load_singularity() {
  local profile_script

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
  if ! command -v singularity >/dev/null 2>&1; then
    module load EB/apps >/dev/null 2>&1 || true
    module load singularity/4.1.5 >/dev/null 2>&1
  fi
}

# Build one node-local source tree with the real inputs overlaid, once per
# allocation. A single bind avoids nested mount points inside a read-only tree.
stage_inputs() {
  local lock="$LOCAL/stage.lock" tree="$SOURCE_TREE" tmp="$SOURCE_TREE.tmp"
  local workspace="$INPUTS_RUN/.state/dvc-workspace" root cache_file

  mkdir -p "$LOCAL"
  (
    flock -x 9
    if [ -f "$tree/.ready" ]; then
      exit 0
    fi
    chmod -R u+w "$tmp" 2>/dev/null || true
    rm -rf "$tmp"
    mkdir -p "$tmp"
    rsync -a "$TEMPLATE_RUN/source/" "$tmp/"
    # Bundle files are read-only; the node-local copy must accept overlays.
    chmod -R u+w "$tmp"
    for root in data/files data/processed; do
      mkdir -p "$tmp/$root"
      rsync -aL "$workspace/$root/" "$tmp/$root/"
    done
    # The HPO base config reads the mobility-aware embeddings by DVC hash.
    cache_file="$DVC_CACHE/files/md5/${REGION_EMBEDDINGS_MD5:0:2}/${REGION_EMBEDDINGS_MD5:2}"
    mkdir -p "$tmp/outputs/region_embeddings"
    cp "$cache_file" "$tmp/outputs/region_embeddings/region_embeddings.pt"
    test -d "$tmp/data/processed/real_with_id.zarr"
    test -d "$tmp/data/files/mobility.zarr"
    touch "$tmp/.ready"
    rm -rf "$tree"
    mv "$tmp" "$tree"
  ) 9>"$lock"
}

# Start the MPS control daemon for the one visible GPU.
start_mps() {
  # Parallel sweeps on one node each run their own daemon for their own GPU.
  MPS_PIPE_DIR="$LOCAL/mps-${GPU_UUID:4:8}-$MODE-k$WORKERS"
  rm -rf "$MPS_PIPE_DIR"
  mkdir -p "$MPS_PIPE_DIR/pipe" "$MPS_PIPE_DIR/log"
  export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR/pipe"
  export CUDA_MPS_LOG_DIRECTORY="$MPS_PIPE_DIR/log"
  if [ -n "$MPS_THREAD_PCT" ]; then
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$MPS_THREAD_PCT"
  fi
  nvidia-cuda-mps-control -d
}

# Stop MPS and keep its logs with the result.
stop_mps() {
  if [ -n "$MPS_PIPE_DIR" ] && [ -d "$MPS_PIPE_DIR/pipe" ]; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    cp -r "$MPS_PIPE_DIR/log" "$OUT/mps-log" 2>/dev/null || true
  fi
}

# Sample the GPU once per second for the whole condition.
start_gpu_sampler() {
  nvidia-smi -i "$GPU_UUID" -l 1 --format=csv,noheader,nounits \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
    >"$OUT/gpu_samples.csv" 2>"$OUT/gpu_sampler.err" &
  SAMPLER_PID=$!
  # PCIe RX/TX in MB/s, to show whether K workers contend for host transfers.
  nvidia-smi dmon -i "$GPU_UUID" -s t -d 1 -o T \
    >"$OUT/pcie_samples.txt" 2>"$OUT/pcie_sampler.err" &
  PCIE_SAMPLER_PID=$!
}

# Stop children on any exit so no worker or daemon outlives the step.
cleanup() {
  local pid

  trap - EXIT
  for pid in "${WORKER_PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
  [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true
  [ -n "$PCIE_SAMPLER_PID" ] && kill "$PCIE_SAMPLER_PID" 2>/dev/null || true
  stop_mps
}

# Launch one HPO worker in a container that shares the host PID and IPC
# namespaces, which MPS credentials require (--containall breaks MPS).
launch_worker() {
  local worker="$1" override
  local worker_dir="$OUT/worker-$worker"
  local threads=$((CPUS / WORKERS))
  local -a args=(exec --cleanenv --contain --nv
    --bind "$SOURCE_TREE:/workspace/source:ro"
    --bind "$OUT:/workspace/run:rw"
    --bind "$worker_dir/tmp:/tmp:rw")

  mkdir -p "$worker_dir/tmp"
  if [ "$MODE" = "mps" ]; then
    args+=(--bind "$MPS_PIPE_DIR/pipe:$MPS_PIPE_DIR/pipe"
      --env "CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR/pipe")
  fi
  args+=(
    --env "PATH=/opt/epiforecaster/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    --env "PYTHONPATH=/workspace/source"
    --env "PYTHONFAULTHANDLER=1"
    --env "PYTHONUNBUFFERED=1"
    --env "XDG_CACHE_HOME=/tmp/.cache"
    --env "MPLCONFIGDIR=/tmp/matplotlib"
    --env "TMPDIR=/tmp"
    --env "EPFORECASTER_STAGE_TO_NVME=0"
    --env "OMP_NUM_THREADS=$threads"
    --env "CUDA_VISIBLE_DEVICES=$GPU_UUID"
    --env "EPIFORECASTER_WORKER_ID=$worker"
    --pwd /workspace/source)
  if [ "$DETERMINISTIC" = "1" ]; then
    args+=(--env "CUBLAS_WORKSPACE_CONFIG=:4096:8")
  fi
  if [ -n "$STUB" ]; then
    args+=(--bind "$STUB:/workspace/stub.py:ro")
  fi
  args+=("$SIF" python)
  if [ "$PROFILE" = "1" ]; then
    args+=(-m cProfile -o "/workspace/run/worker-$worker.prof")
  fi
  if [ -n "$STUB" ]; then
    args+=(/workspace/stub.py)
  elif [ "$DETERMINISTIC" = "1" ] || [ "$EXACT_KERNELS" = "1" ]; then
    # Configure torch in-process before the worker module imports anything.
    local prelude='import runpy, sys, torch'
    if [ "$DETERMINISTIC" = "1" ]; then
      prelude+='
torch.use_deterministic_algorithms(True)
print("deterministic algorithms on", flush=True)'
    fi
    if [ "$EXACT_KERNELS" = "1" ]; then
      # Inductor picks some Triton configs by timing, and the configs reduce
      # in different orders; one config per kernel removes that choice.
      # Private API, checked against torch 2.9.
      prelude+='
import torch._inductor.config as ic
import torch._inductor.runtime.triton_heuristics as th
ic.compile_threads = 1
_autotune = th.cached_autotune
th.cached_autotune = lambda hints, configs, *a, **k: _autotune(hints, configs[:1], *a, **k)
print("exact kernels on", flush=True)'
    fi
    prelude+='
sys.argv[0] = "hpsearch_worker"
runpy.run_module("scripts.hpo.hpsearch_worker", run_name="__main__", alter_sys=True)'
    args+=(-c "$prelude")
  else
    args+=(-m scripts.hpo.hpsearch_worker)
  fi
  args+=(
    --config configs/production_only/train_epifor_mn5_full.yaml
    --study-name "$STUDY_PREFIX-$MODE-k$WORKERS"
    --journal-file /workspace/run/study.journal
    --run-root /workspace/run/trials
    --n-trials "$TRIALS" --epochs "$EPOCHS" --pruning-start-epoch 999
    --sampler random --search-space lr_only --seed 42
    # Pin the compute-heavy trial shape so every worker does equal work.
    --override data.mobility_min_flow_raw=1.0
    --override data.sample_ordering=node
    --override model.gnn_depth=2
    --override model.gnn_module=gat
    --override model.mobility_embedding_dim=128
    --override model.observation_heads.delta_forecasting=true
    --override model.observation_heads.residual_hidden_dim=64
    --override training.weight_decay=1.4258706676899107e-07
    --override "training.run_final_evaluation=$([ "$PROFILE" = "1" ] && echo true || echo false)"
    --override training.progress_log_frequency=10
  )
  if [ -n "$COMPILE_MODE" ]; then
    args+=(--override "training.compile_mode=$COMPILE_MODE")
  fi
  for override in "${EXTRA_OVERRIDES[@]}"; do
    args+=(--override "$override")
  done
  local -a launcher=(timeout -k 30 "$CONDITION_TIMEOUT")
  if [ -n "$CPU_LIST" ]; then
    launcher+=(taskset -c "$CPU_LIST")
  fi
  "${launcher[@]}" singularity "${args[@]}" >"$OUT/worker-$worker.log" 2>&1 &
  WORKER_PIDS+=("$!")
}

# Run one condition: stage, optionally start MPS, run K trials, record status.
run_condition() {
  local worker status failed=0 started finished

  MODE="${CONDITION%%:*}"
  WORKERS="${CONDITION##*:}"
  if [[ ! "$MODE" =~ ^(plain|mps)$ ]] || [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid --condition: $CONDITION" >&2
    return 2
  fi
  if [ $((CPUS % WORKERS)) -ne 0 ]; then
    echo "--cpus ($CPUS) must divide evenly by K ($WORKERS)" >&2
    return 2
  fi
  LOCAL="${TMPDIR:-/scratch/tmp/${SLURM_JOB_ID:?}}/mps-probe"
  SOURCE_TREE="$LOCAL/src-$(basename "$TEMPLATE_RUN")"
  OUT="$PROBE_ROOT/results/$MODE-k$WORKERS"
  mkdir "$OUT"

  load_singularity
  stage_inputs
  trap cleanup EXIT
  # Both nvidia-cuda-mps-control and CUDA clients need the healthy GPU only.
  export CUDA_VISIBLE_DEVICES="$GPU_UUID"
  if [ "$MODE" = "mps" ]; then
    start_mps
  fi
  {
    echo "node=$(hostname)"
    echo "loadavg_start=$(cut -d' ' -f1-3 /proc/loadavg)"
    echo "mem_free_gb_start=$(free -g | awk '/^Mem:/ {print $7}')"
  } >"$OUT/host.txt"
  start_gpu_sampler
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for ((worker = 0; worker < WORKERS; worker++)); do
    launch_worker "$worker"
  done
  for worker in "${!WORKER_PIDS[@]}"; do
    if wait "${WORKER_PIDS[$worker]}"; then
      status=0
    else
      status=$?
      failed=1
    fi
    echo "worker_$worker=$status" >>"$OUT/host.txt"
  done
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "loadavg_end=$(cut -d' ' -f1-3 /proc/loadavg)" >>"$OUT/host.txt"
  printf '{"mode":"%s","workers":%s,"cpus":%s,"cpu_list":"%s","epochs":%s,"mps_thread_pct":"%s","compile_mode":"%s","started":"%s","finished":"%s","failed":%s,"gpu_uuid":"%s","slurm_job":"%s","slurm_step":"%s"}\n' \
    "$MODE" "$WORKERS" "$CPUS" "$CPU_LIST" "$EPOCHS" "$MPS_THREAD_PCT" \
    "${COMPILE_MODE:-config}" "$started" "$finished" \
    "$failed" "$GPU_UUID" "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" >"$OUT/condition.json"
  return "$failed"
}

main() {
  parse_arguments "$@"
  validate_arguments
  case "$MODE_COMMAND" in
    sweep) run_sweep ;;
    condition) run_condition ;;
    *) usage >&2; return 2 ;;
  esac
}

main "$@"

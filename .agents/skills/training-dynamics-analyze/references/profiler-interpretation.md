# Profiler and Timing Interpretation

## Contents

- Evidence boundary
- Trace review
- Bottleneck classification
- Common confounders

## Evidence boundary

PyTorch profiler traces cover configured active windows. Record the number of
traces, profiler-step events, configured wait/warmup/active schedule, CPU/CUDA
activities, memory recording, and stack recording when available.

Never extrapolate two profiled steps into whole-run throughput. Prefer persisted
training timing for run-level distributions and use traces to explain selected
steps.

Profiler operation durations are often nested. Inclusive durations may overlap;
do not add them and call the result wall time or utilization.

## Trace review

For each trace:

- identify profiler step boundaries and their durations;
- separate CPU operations, CUDA kernels, runtime calls, copies, synchronization,
  and compilation;
- list hotspots by both total duration and count;
- note whether memory data or stack traces were recorded;
- compare traces rather than merging compilation-heavy and steady-state windows.

Long operator names should remain traceable to their exact profiler names even
when abbreviated in a figure.

## Bottleneck classification

Use multiple signals:

- **Input-bound candidate:** elevated data-loading time is a substantial part of
  sampled batch time and corresponds with lower throughput.
- **CPU launch-bound candidate:** many short kernels with substantial runtime or
  launch activity and limited kernel duration.
- **GPU compute-bound candidate:** steady-state steps are dominated by a small
  number of long kernels without excessive synchronization.
- **Synchronization-bound candidate:** repeated or long synchronization events
  serialize otherwise asynchronous work.
- **Compilation-dominated trace:** compile/Inductor/CompiledFxGraph activity
  occupies the profiled window.
- **Transfer-bound candidate:** copies consume material sampled time and align
  with device stalls.

Call these candidates unless a representative steady-state profile and
supporting run-level timing establish the conclusion.

## Common confounders

- First profiled epochs may include graph compilation and cache population.
- Profiler overhead changes timing; compare profiled and unprofiled throughput
  only with controlled runs.
- Data loading can occur outside the traced model region.
- CUDA kernels are asynchronous; CPU duration is not GPU execution duration.
- Memory conclusions require `record_memory=true`.
- Stack attribution requires `with_stack=true`.
- Concurrent work in a shared interactive allocation invalidates isolated
  throughput claims unless resource exclusivity is verified.

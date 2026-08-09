# SIRHD Calibration Experiment Plan (Revised)

## Prior work on this branch

### Already implemented (commit 3776af3)
- Destination-dated `latent_death_flow_true` target (replaces cumulative-D loss)
- Three latent scaling modes: `none`, `std`, `asinh`
- Per-component loss tracking (raw, transformed, weighted)
- Training-split scale fitter with immutable `latent_scale_receipt.json`
- Latent gradient telemetry sidecar (`pathway_gradient_frequency`)
- Oracle-X0 and teacher-forced one-step diagnostic protocols
- Pathway metrics: per-compartment MAE/RMSE, mass conservation, death-transition error
- Three frozen MN5 workflow configs: `mn5_latent_rescaling_{unscaled,std,asinh}_mse.yaml`

### Evidence from prior runs
- **Old corpus (e2d683c, pre-refactor SIRD)**: `sir_param_sq_norm_share = 3.4e-5` — 
  SIR parameters received ~0.003% of total gradient energy. Latent supervision was 
  effectively dead.
- **Smoke run (44404125, post-refactor, 5-run, 3 epochs)**: Raw latent losses confirm 
  the fraction-space imbalance:
  - S=0.79, R=0.79 (dominate) → I=1.1e-5 → H=3.4e-7 → death_flow=1.7e-11 (negligible)
  - Total weighted (w=0.01, scaling=none): 0.003 — invisible vs WW NLL ~2333
  - Gradient telemetry was NOT enabled (`pathway_gradient_frequency=0`)

### What has NOT been tested
- The three scaling modes (`none`, `std`, `asinh`) have never run on real data
- Per-task latent gradient shares have never been measured with the new code
- Oracle-X0 and teacher-forced diagnostics have never produced evidence
- `std` scaling (which divides by per-compartment std) has never been validated 
  against the actual scale receipt values

## Revised experiment matrix (3 arms + 1 baseline, 3 epochs each)

The original plan proposed 5 arms with 10 epochs. This is trimmed because:
- Arm A (no latent supervision) is informative but not blocking — we already know 
  the observation metrics from the smoke run.
- Arm D (MSE observations) conflates two variables (loss type + scaling) — violates 
  the matched-study contract from the gradient recovery plan.
- 10 epochs is excessive for a calibration check; 3 epochs at ~10 min/epoch on 
  the 2-run corpus is sufficient to measure gradient shares and latent RMSE trends.

| Arm | Name | Scaling | Weight | Telemetry | Purpose |
|-----|------|---------|--------|-----------|---------|
| **0** | baseline | none | 0.0 | yes | Observation-only; establishes latent RMSE floor |
| **1** | unscaled | none | 0.01 | yes | Current default; confirms gradient starvation |
| **2** | std | std | 0.01 | yes | Primary candidate; normalizes compartment scales |
| **3** | asinh | asinh | 0.01 | yes | Robust transform; compresses I/H/death_flow outliers |

All arms keep `total_weight=0.01` and fixed per-task coefficients `0.002` each, 
matching the gradient recovery plan's matched-study contract. The ONLY variable 
is the latent transform mode. This is a clean single-variable study.

### Shared configuration (from gradient recovery plan)

```
training:
  epochs: 3
  pathway_gradient_frequency: 100     # REQUIRED: enables latent gradient sidecar
  gradient_snapshot_frequency: 0
  gradient_clip_value: 5.0

output:
  write_granular_eval: true           # REQUIRED: enables latent pathway metrics

model.params.sir_physics:
  residual_clip: 10000.0              # keep production value for matched comparison

# Fixed from study contract:
training.loss.joint.latent:
  total_weight: 0.01
  scaling: <arm-specific>
  require_complete_targets: true
```

### Why keep total_weight=0.01 instead of increasing it?

The gradient recovery plan explicitly states: "Keep the total fixed latent 
coefficient at 0.01." The study isolates the **transform** effect, not the weight.
If `std` scaling makes the gradient share meaningful at w=0.01, then a subsequent 
study can tune the weight. Changing both at once violates the matched contract.

## What each arm tells us

**Arm 0 (baseline)**: What does the model learn about latent compartments with 
ZERO explicit supervision? The observation loss still backprops through the SIR 
roll-forward, so latent RMSE won't be random — but it won't be calibrated either.

**Arm 1 (unscaled)**: Confirms the gradient starvation quantitatively. We expect:
- `latentgrad_combined_squared_gradient_share` ≈ 1e-5 (matching the old evidence)
- S and R dominate the latent gradient; I/H/death_flow get ~0

**Arm 2 (std)**: The primary test. With std normalization:
- S std ≈ 0.25, I std ≈ 0.023, H std ≈ 0.0016, R std ≈ 0.28, death_flow std ≈ 2.3e-5
- After dividing by std: all compartments contribute comparably to the loss
- Expected: 10-100× increase in I/H/death_flow gradient contribution

**Arm 3 (asinh)**: Tests whether the asinh compressive transform further improves 
I/H/death_flow calibration by reducing outlier sensitivity. Only S/R use plain 
std division; I/H/death_flow get `asinh(x/std)`.

## Primary metrics (per arm)

1. **Gradient share** (from training_dynamics.jsonl, median across logged steps):
   - `latentgrad_combined_squared_gradient_share`
   - Per-task norms: S, I, H, R, death_flow
   - Cosine with observation gradient

2. **Latent accuracy** (from granular eval pathway metrics):
   - `latent_mae_{s,i,h,r,d}_fraction` — per-compartment
   - `latent_rmse_{s,i,h,r,d}_fraction`
   - `latent_mae_{name}_horizon_{NN}_fraction` — per-horizon degradation

3. **Structural SIRHD validity**:
   - `latent_model_sirhd_mass_error_from_one_mae`
   - `latent_death_transition_mae`
   - `latent_physics_residual_mae`

4. **Observation guardrails** (should not degrade >5% vs arm 0):
   - `r2_{cases,hosp,deaths}_log1p_per_100k`
   - `mae_{cases,hosp,deaths}_log1p_per_100k`

## Decision criteria (from LATENT_RESCALING_V1_STUDY_CONTRACT.md)

A scaled arm (2 or 3) passes only when vs arm 1 (unscaled):
- Median combined latent squared-gradient share improves ≥10×
- Geo-mean abs log error of I/H/death-flow pred-to-target std ratios improves ≥20%
- Mean teacher-forced one-step RMSE across I/H/death_flow improves ≥10%
- Mean oracle-X0 RMSE across I/H/death_flow does not degrade >2%
- Mean free-rollout RMSE across I/H/death_flow does not degrade >5%
- S/R free-rollout RMSE not worse than 5% relative or 1e-4 absolute
- Mean observation MAE across cases/hosp/deaths/WW not worse than 5%
- No finite/nonnegative, conservation, or complete-target contract fails

If neither scaled arm passes, retain unscaled and use evidence for next intervention.
If one passes, select it. If both pass, pick higher gradient share; break ties <5% 
in favor of lower teacher-forced RMSE, then `std` (simpler).

## Execution plan

1. Create 4 training configs derived from `train_epifor_smoke_netcdf.yaml`:
   - Each sets `pathway_gradient_frequency=100` and `write_granular_eval=true`
   - Arm-specific: `total_weight` and `scaling` override
2. Preprocess the 2-run smoke corpus (already done: `synthetic_netcdf_smoke_5run.nc` 
   from run 44402878 — can re-use for the 2-run c3d4810 corpus)
3. Run all 4 sequentially on an interactive ACC allocation (~30 min each)
4. Compare via `scripts/analysis/analyze_training_dynamics.py` and a new comparison table
5. Total wall time: ~2.5 hours for all 4 arms

## Estimated time

- 4 arms × 3 epochs × ~10 min/epoch = ~2 hours
- Preprocessing (if needed): ~2 min for 2 runs
- Evaluation: ~2 min per arm
- Total: ~2.5 hours

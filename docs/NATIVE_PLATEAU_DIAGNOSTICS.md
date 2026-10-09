# Native plateau diagnostic experiment

The reference is the SIR/mobility arm of
`native-sir-mobility-full10ep-60cffe4-20261003-a`. Its best validation epoch is 2.
Training wastewater CRPS falls from 0.9287 to 0.6948 between epochs 2 and 10;
validation wastewater CRPS rises from 0.8703 to 0.9781. The validation set has
two adjacent forecast origins and 22 distinct exact wastewater source/date
pairs. These observations motivate the experiment; they do not identify a cause.

Use `workflow/config/mn5_native_plateau_diagnostics.yaml` with the immutable MN5
launcher. Retain the original run. The new run keeps the model, learning rate,
cosine scheduler, batch size, seed, input window, horizon and corrected input
identities. It reserves 30% of eligible origins for validation and 10% for test.
The 28-day purge remains in force. Construction requires at least 56 effective
validation origins. Source sets are regenerated from the declared new policy;
their identities must be compared with the reference and reported. Results are
from a new split, rather than a replication of the original endpoint comparison.

Each of ten full epochs records:

- canonical validation metrics and predictions, with interval coverage and width;
- metrics from a fixed training panel of up to three eligible origins per
  head/source, spaced over its training interval and evaluated without dropout;
- canonical source/horizon and 28-day calendar-origin-block metrics;
- training objective, active-head support, actual update learning rate, total
  pre-clip gradient norm, and sampled disjoint parameter-group norms.

The panel identity and epoch exports have SHA-256 records in the checkpoint and
terminal dynamics manifest. Diagnostics do not affect checkpoint selection.
The extra training-panel evaluation preserves the training RNG state. Component
norms are gradient allocation measurements, not task conflict or feature importance.
The heterogeneous spatial encoder is labelled separately from a mobility-only GNN.

Inspect canonical interval coverage, WIS components, per-source scores and
calendar-origin-block scores before selecting an intervention. Overlapping
windows, assays from one plant and adjacent calendar blocks are not independent
replicates. Preserve equal-head validation selection and use the test set only
for the final selected checkpoint, as required by the workflow.

If deterioration recurs across sources and calendar blocks while fixed-panel
training scores improve, test stronger regularization or a lower learning rate.
If it is concentrated in a plant or period, audit that source and its scaling,
observation mask and distribution before changing optimization. If interval
width falls while error penalties rise, test the uncertainty model and its
calibration. Gradient conflict needs a separate targeted probe if head-level
evidence supports it. Compare any intervention with this baseline on the same
new split and update budget. A change in total epochs also changes the cosine
schedule and must not be treated as an epoch-only intervention.

## SIR mobility campaign with 50 epochs

The current extended campaign is
`configs/campaigns/native_sir_mobility_50ep_v3_v1.yaml`. Run it through
`workflow/config/mn5_native_sir_mobility_50ep_v3.yaml`. It has two identities:
SIR with mobility on and SIR with mobility off. Each uses spatial fold 0 of
the ten-fold source partition and model seed 42. Each trains for 50 full
epochs from initialization, with no early stopping or batch cap.

The clinical input is `data/processed/real_with_id_v3.zarr`, pinned by DVC
directory hash `17bb89b338bb0314b2f90d7dfd9dd1c8.dir`. It retains the causal
clinical filter and missing-data policy from v2 and removes municipal wastewater.
Native EDAR observations use the separate corrected wastewater input. Source
assignments are generated from v3 and must agree between mobility conditions.
The v3 input changes preprocessing relative to the ten-epoch reference, so this
comparison does not isolate the epoch budget. The old-input 50-epoch declarations
and immutable run remain available as records of the superseded campaign.

Both conditions use the broader validation policy and all diagnostics described
above. Mobility is the sole declared difference between their hydrated model
configurations. Their generated source assignments must agree. The cosine
learning-rate schedule spans 50 epochs. The ten-epoch acceptance run remains
separate and retains its original schedule and artifacts.

The workflow requests two GPU workers on `acc_bscls`, with 20 CPUs per worker
and an eight-hour allocation limit. It stops claiming new trials 30 minutes
before closeout. This limit is a resource ceiling, not an estimate of completion
time. Results have one fold and one seed per condition; they do not establish
variation across folds or initialization seeds.

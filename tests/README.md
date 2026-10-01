# Test suite scope

Use bounded integration tests as the local check before an MN5 run. These tests
must exercise the same configuration, dataset, model, training, checkpoint, and
evaluation paths as the cluster workflow. Check the generated evaluation files and
their contents. Add a new integration case when a feature changes these paths.

Keep isolated tests for contracts that the integration tests cannot check well:
configuration parsing and validation, numerical edge cases, data alignment,
sampling invariants, and provenance or artifact integrity. Group related cases
in one deterministic test when they share the same setup and failure meaning.

Remove tests that only repeat constructor calls, tensor shapes, internal helper
calls, or plot rendering already covered by a higher-level path. Do not replace
a specific scientific invariant with a generic smoke assertion. Keep the
integration suite bounded for local use; substantive training remains on MN5.

The native heterogeneous pipeline is covered by `test_native_pipeline.py`.
Vaccination is enabled in every case. The cohort covers mobility on and off,
strict mode on and off, and a clinical-only run. One paired run compares the real
compiler on and off, with the other feature flags fixed. Poisoned future labels
are processed again and must leave split balancing, every training batch,
training loss, model weights, and pre-origin predictions unchanged. The poison
must also cause a large increase in held-out WIS. Repeated CLI evaluation checks
native IDs, censor exclusion, signed excess deaths, and all output formats.

Keep focused tests for weekly calendar and mask arithmetic, loss gradients,
causal inputs, config parsing, cache integrity, and checkpoint selection/resume.

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

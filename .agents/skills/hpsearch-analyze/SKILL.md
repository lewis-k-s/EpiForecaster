---
name: hpsearch-analyze
description: Analyze EpiForecaster Optuna HPO studies through JournalStorage with exact trial identities and states, immutable-run receipt reconciliation, parameter coverage, trial dynamics integrity, and optional seeded fANOVA importance. Use for live, partial, or completed HPO audits and search-space evidence review.
---

# HPO Analysis

Prefer the immutable workflow run root:

```bash
uv run --no-sync hpsearch-analyze --run-root <run-root>
uv run --no-sync hpsearch-analyze --run-root <run-root> --top 10 --text
```

For a standalone journal, require the exact study name:

```bash
uv run --no-sync hpsearch-analyze \
  --journal <study.journal> \
  --study-name <study-name>
```

JSON is the default. Use `--compact` for compact JSON and `--text` for a concise
human-readable view.

## Workflow

1. Load the study through Optuna `JournalStorage`; never parse journal records
   manually.
2. Preserve Optuna trial numbers, directions, distributions, user attributes,
   and every state, including parameterless failures and pruned/running trials.
3. For `--run-root`, reconcile the journal with study summary, selection,
   evaluation, promotion, cleanup, and workflow receipts.
4. Review parameter coverage before comparing conditional parameters.
5. Use `--importance` only when at least 20 trials completed. The analyzer uses
   seeded fANOVA over parameters common to all completed trials.

The analyzer is descriptive. It does not automatically narrow numeric ranges or
remove categorical values. Treat a selected best trial as provisional until the
workflow's independent evaluation and promotion evidence is complete.

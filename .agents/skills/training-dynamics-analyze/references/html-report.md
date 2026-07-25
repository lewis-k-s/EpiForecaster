# HTML Report Composition

## Required structure

Author one `report.html` with:

1. title, run identity, status, step range, and artifact hash;
2. executive assessment;
3. evidence coverage table;
4. loss behavior;
5. gradient and clipping behavior;
6. adaptive GradNorm and gradient snapshots when available;
7. timing and profiler behavior;
8. cross-evidence findings;
9. limitations;
10. bounded next checks;
11. provenance and evidence lifecycle.

Use the bundled template for document structure and CSS. Remove sections that
have no corresponding evidence only after recording the missing evidence in the
coverage table.

## Evidence integration

Use relative image paths from `bundle_manifest.json`. Include the loss curve,
gradient/clipping plot, component-gradient plot, timing plot, and profiler plot
when present.

Render compact HTML tables for decision-relevant values. Do not dump full CSVs
into the page. Prefer:

- evidence status;
- first/final/best loss and spike steps;
- clipping count, fraction, and strongest factor;
- component median and p95 squared-norm shares;
- timing median, p95, and maximum;
- profiler-step durations and leading operations;
- GradNorm task-weight ranges when available.

Use `<details>` for provenance, long operator names, or secondary metrics.

Build provenance tables from `tables/provenance.csv`. Show DVC pointer paths and
DVC hashes separately from SHA-256 evidence hashes. State the recorded
`retention_class` and `dvc_tracking` value without upgrading run-scoped evidence
to durable DVC evidence in prose.

## Qualitative writing

Write conclusions only after inspecting the plots and relevant tables. For each
material finding, include:

- observation;
- interpretation;
- alternative explanation or limitation;
- bounded follow-up check.

Reference exact steps, epochs, traces, or table values. Coded analyzer signals
identify associations to review; translate them into prose only after checking
the underlying evidence.

Do not describe:

- training-only behavior as generalization;
- component gradient share as feature importance;
- correlation or step coincidence as causation;
- inclusive profiler duration as additive wall time;
- a short active profiler window as whole-run throughput.

## HTML requirements

- Use semantic headings, figures, tables, and `<figcaption>`.
- Include descriptive `alt` text for every image.
- Keep CSS inline in `<style>` and JavaScript unnecessary.
- Use UTF-8 and a responsive viewport.
- Keep the report self-contained except for relative PNG assets.
- Escape source paths and other evidence-derived text used in HTML.
- Do not leave template tokens matching `{{...}}`.

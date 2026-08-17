# EpiForecaster project context

This repo defines a forecasting pipeline for time series data and variants of the model which include mobility and geospatial embeddings.

We model an epidemiological task to forecast disease incidence and prevalence metrics.

## Mission & Safety Scope

- Mission: build epidemiological forecasting tools for public-health situational awareness, early outbreak monitoring, and planning support.
- Model framing: this project uses a transformer-based forecaster with physics-grounded SIR roll-forward dynamics, plus optional mobility/geospatial context, to handle sparse multivariate signals and predict public-health outcomes.
- Intended use: forecasting, uncertainty analysis, retrospective evaluation, and operational decision support for public-health officials and researchers.
- Out of scope: any request that enables pathogen design, biological optimization, lab protocol development, transmission enhancement, or operational misuse.
- Agent behavior: if a task drifts into out-of-scope bio guidance, decline that part and redirect to safe, high-level epidemiological modeling and public-health analytics.

For detailed information on the forecaster model design see @EPIFORECASTER.md
For detailed information on the region embedding model design see @REGION2VEC.md

## Project Structure & Module Organization

- Source entry point: `cli.py` (Click-based CLI)
- Core packages: `models/` (architectures), `graph/` (graph building), `training/` (unified trainer), `data/` (dataset management), `utils/` (helpers).
- Configurations: `configs/` (YAML configuration templates and examples).
- Data and outputs: `data/` (inputs, preprocessing), `outputs/` (artifacts, logs), `reports/` (figures/notes). Do not commit large files.
- Tests: `tests/` (pytest). Some legacy tests exist at repo root; prefer `tests/`.

## Development Environment

```bash
# Install dependencies
uv sync

# Dev loop (run these often)
uv run ruff check .        # linting (required)
uv run pytest tests/       # tests (required)
uv run ruff format .       # formatting
uv run pyright .           # type checking (best-effort; excludes dataviz/plotting)

# CLI usage
uv run main --help  # entrypoint cli.py
```

## Coding Style & Naming Conventions

- Python 3.11+. Format with Black (88 chars). Lint with Ruff; type-check with Pyright.
- Indentation: 4 spaces; no tabs. Imports sorted (Ruff isort rules).
- Naming: modules `snake_case.py`, classes `PascalCase`, functions/vars `snake_case`, constants `UPPER_SNAKE_CASE`.
- Prefer type hints, dataclasses where appropriate, and small, pure functions.
- Tensor ops: prefer `einops` (`rearrange`, `reduce`, `repeat`) for matrix/tensor manipulation over manual `view/reshape/permute` for clarity.
- For Python types, prefer native list, dict, tuple instead of the typing library. Prefer the `x | None` notation over Optional
- For all model and training config or feature flags, set values and defaults in our config dataclasses and use direct class attribute access instead of getattr. Config value defaults are defined on during yaml parsing, runtime fallbacks are undesirable in most cases.

## Testing Guidelines

- Framework: pytest. Place tests under `tests/` as `test_*.py`.
- Cover new/changed logic with focused unit tests; avoid network/IO in unit tests.
- Use fixtures for sample data; keep tests deterministic (seed RNGs).
- Quick examples: `uv run pytest -k graph tests/`, `uv run pytest -q tests/`.

### Core Testing Boundary

- Strong automated coverage is required for the primary training and evaluation pipeline, from configuration and data ingestion through evaluation artifact generation. This includes:
  - parsing YAML and CLI overrides into the canonical config dataclasses;
  - data loading, preprocessing, masking, scaling, splitting, sampling, and batching;
  - model construction and forward passes;
  - every supported feature flag in both its enabled and disabled states, including interactions that materially change model inputs or outputs;
  - training losses, optimization behavior, checkpoint save/load/resume contracts, and evaluation;
  - evaluation metrics, aggregations, selections, metadata, and generation of every machine-readable evaluation output file.
- The strongly tested core ends at the evaluation outputs. Plotting, rendering, report layout, and exploratory visualization code are outside the automated coverage target; do not add exhaustive plot tests or pursue coverage percentages for those modules.
- Plotting and visualization code must consume canonical evaluation outputs. It must not independently re-derive metrics, indicators, masks, sparsity measures, window selections, aggregations, or other evaluation results.
- If a visualization needs a new indicator or derived value, implement and test it in the evaluation pipeline, include it in the canonical evaluation outputs, and keep the plotting layer limited to loading and presentation.
- Keep evaluation computation separate from plotting so the tested core can be executed without graphical dependencies. Exploratory analysis remains outside the core coverage target unless it is promoted into the canonical training or evaluation pipeline.

Tests are organized with markers for selective execution:

- `@pytest.mark.region`: Tests related to region2vec, region losses, spatial autocorrelation
- `@pytest.mark.epiforecaster`: Tests related to the main epidemiological forecaster model

Usage:

```bash
uv run pytest -m region -v
uv run pytest -m epiforecaster -v
uv run pytest -m "not region" -v
uv run pytest --markers
```

## Training & Dataset Workflow

For config-driven training and the dataset pipeline, load the **`model-training`** skill.

Key local entry points:

- `uv run train regions --config configs/train_regions.yaml` — local Region2Vec training
- `uv run train epiforecaster --config configs/train_epifor_smoke.yaml` — bounded local acceptance only
- `uv run preprocess (regions|epiforecaster)` — data preparation

Substantive EpiForecaster training and evaluation use the immutable MN5
Snakemake workflow. Do not introduce additional local EpiForecaster training
configs as alternate production interfaces.

### Remote MN5 Cluster Work

All remote model operations use `scripts/workflow/mn5_workflow.sh` and are
governed by the **`mn5-cluster-usage`** skill. **Load that skill** before remote
preparation, transfer, submission, monitoring, or artifact inspection; it
defines endpoint and allocation policy, while this repository's launcher and
`docs/MN5_SNAKEMAKE.md` define the immutable workflow contract.

> **WARNING**: Do not run configs in `configs/production_only/` locally. They require GPU cluster resources. See `configs/production_only/README.md`.

### Immutable Deployment and Artifact Model

Do not assume a GPFS project or run directory is a Git checkout. Cluster
execution separates:

1. the authoritative local Git checkout where source changes are developed;
2. an immutable deployed source bundle used by a particular cluster run;
3. a versioned run root containing configuration, manifests, logs, seeds, and
   execution state;
4. generated scientific artifacts such as checkpoints and Zarr datasets; and
5. downstream DVC pointers and materializations.

For an existing run, treat its manifest and deployed-bundle hashes as the
authority for the code and inputs that produced its artifacts. A local checkout
is authoritative for new development, but it must not be assumed to match an
old deployment merely because paths or filenames agree. Reconcile Git commit,
source-bundle hash, configuration, container or environment identity, seeds,
input identities, and artifact hashes where available. If the run lacks enough
metadata to establish this identity, report a provenance limitation rather
than inferring one.

Never patch or overwrite an immutable deployed bundle, run root, or generated
artifact in place. Make changes in a Git branch, validate them, then create a
new immutable bundle and versioned run. Preserve the existing run and record
the new code, configuration, environment, seed, input, and output identities.

## Paper Workflow

- Manuscript workspace overview: `tex/README.md`
- Main manuscript source: `tex/EpiForecaster/sn-article.tex`
- Paper build and clean workflow: `tex/EpiForecaster/WORKFLOW.md`
- Manuscript helper functions: `tex/spells.sh`
- When editing paper sources or figure references, review the TeX workspace context files first, especially `tex/README.md` and `tex/EpiForecaster/WORKFLOW.md`.
- After any manuscript edit, build the paper before finishing by sourcing `tex/spells.sh` and running `compile_paper`.

## Commit & Pull Request Guidelines

- Commits: imperative mood, scoped, and small (e.g., `graph: add OD edge builder`). Reference issues when applicable.
- PRs must include: concise description, rationale, screenshots of plots (if any) saved to `outputs/`, and notes on breaking changes.
- CI gates locally: ensure `uv run ruff check .` and `uv run pytest tests/` pass before requesting review.

## Agent-Specific Instructions

- `git checkout` is denied by permissions. Use `git switch` for branch switching and `git restore` for discarding file changes instead.
- Do not chain git commands with `&&` (e.g. `git add ... && git commit ...`). The permissions system matches against the full command string, so chaining breaks pattern matching. Use separate bash calls instead.
- New features and refactors are assumed to be 'cutover' implementations, not preserving backwards compatibility with previous behavior, unless specified otherwise.
- Prefer `uv run` for project test suites, CLI commands, scripts, and workflow execution; update `pyproject.toml` if adding deps.
- Do not modify `data/` inputs or commit large `outputs/`. Place generated artifacts in `outputs/` and ignore by default.
- Keep patches minimal and consistent with existing style; avoid unrelated refactors. If heavy compute is required, add flags and default to lightweight settings.
- Use configuration-driven development: copy and modify templates from the `configs/` directory or provide CLI level overrides with dot-path syntax like `--override model.strict=true` rather than hardcoding parameters.
- **Zarr Data Access**: Always use `xarray.open_zarr()` instead of `zarr.open()` for inspecting or loading zarr datasets. Xarray provides labeled dimensions, coordinates, and better integration with the data pipeline.

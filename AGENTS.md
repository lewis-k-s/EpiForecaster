# Repository Guidelines

This repo defines a forecasting pipeline for time series data and variants of the model which include mobility and geospatial embeddings.

We model an epidemiological task to forecast disease incidence and prevalence metrics.

## Mission & Safety Scope

- Mission: build epidemiological forecasting tools for public-health situational awareness, early outbreak monitoring, and planning support.
- Model framing: this project uses a transformer-based forecaster with physics-grounded SIR roll-forward dynamics, plus optional mobility/geospatial context, to handle sparse multivariate signals and predict public-health outcomes.
- Intended use: forecasting, uncertainty analysis, retrospective evaluation, and operational decision support for public-health officials and researchers.
- Out of scope: any request that enables pathogen design, biological optimization, lab protocol development, transmission enhancement, or operational misuse.
- Agent behavior: if a task drifts into out-of-scope bio guidance, decline that part and redirect to safe, high-level epidemiological modeling and public-health analytics.

For detailed information on the forecaster model design see EPIFORECASTER.md
For detailed information on the region embedding model design see REGION2VEC.md

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
.venv/bin/python -m cli --help  # access CLI commands
# or after installation: epiforecaster --help
```

## Coding Style & Naming Conventions

- Python 3.11+. Format with Black (88 chars). Lint with Ruff; type-check with Pyright.
- Indentation: 4 spaces; no tabs. Imports sorted (Ruff isort rules).
- Naming: modules `snake_case.py`, classes `PascalCase`, functions/vars `snake_case`, constants `UPPER_SNAKE_CASE`.
- Prefer type hints, dataclasses where appropriate, and small, pure functions.
- Tensor ops: prefer `einops` (`rearrange`, `reduce`, `repeat`) for matrix/tensor manipulation over manual `view/reshape/permute` for clarity.
- For Python types, prefer native list, dict, tuple instead of the typing library. Prefer the `x | None` notation over Optional

## Testing Guidelines

- Framework: pytest. Place tests under `tests/` as `test_*.py`.
- Cover new/changed logic with focused unit tests; avoid network/IO in unit tests.
- Use fixtures for sample data; keep tests deterministic (seed RNGs).
- Quick examples: `uv run pytest -k graph tests/`, `uv run pytest -q tests/`.

### Pytest Markers

Tests are organized with markers for selective execution:

- `@pytest.mark.region`: Tests related to region2vec, region losses, spatial autocorrelation
- `@pytest.mark.epiforecaster`: Tests related to the main epidemiological forecaster model

Usage:

```bash
# Run only region tests
uv run pytest -m region -v

# Run only epiforecaster tests
uv run pytest -m epiforecaster -v

# Run all tests except region
uv run pytest -m "not region" -v

# List all markers
uv run pytest --markers
```

## Configuration-Driven Development

This project uses YAML configuration files for reproducible experiments:

- Copy and modify configuration templates from the `configs/` directory
- Preprocessing configs: specify data sources and processing parameters
- Training configs: define model type, hyperparameters, and training settings
- All model variants (base, regions, mobility, full) use the same interface
- Config parsing goes through Python dataclasses (see `training/epiforecaster_trainer.py`, `training/region_embedder_trainer.py`); keep the YAML schema and the dataclass fields in sync and add/adjust docstrings when new config keys are introduced.

## Dataset Workflow

The canonical data pipeline follows: raw data → preprocessing → Zarr datasets → training

- Use `uv run preprocess (regions|epiforecaster)` to convert raw data to canonical format
- Manage datasets with `epiforecaster list-datasets` and `epiforecaster info`
- Use `uv run train (regions|epiforecaster)` to train on the prepared dataset.

## Commit & Pull Request Guidelines

- Commits: imperative mood, scoped, and small (e.g., `graph: add OD edge builder`). Reference issues when applicable.
- PRs must include: concise description, rationale, screenshots of plots (if any) saved to `outputs/`, and notes on breaking changes.
- CI gates locally: ensure `uv run ruff check .` and `uv run pytest tests/` pass before requesting review.

## Agent-Specific Instructions

- New features and refactors are assumed to be 'cutover' implementations, not preserving backwards compatibility with previous behavior, unless specified otherwise.
- Prefer `uv run` for execution and CLI commands over ad-hoc commands; update `pyproject.toml` if adding deps.
- Do not modify `data/` inputs or commit large `outputs/`. Place generated artifacts in `outputs/` and ignore by default.
- Keep patches minimal and consistent with existing style; avoid unrelated refactors. If heavy compute is required, add flags and default to lightweight settings.
- This project is a work in progress. It is ok to make breaking changes as the interface evolves.
- Use configuration-driven development: copy and modify templates from the `configs/` directory rather than hardcoding parameters.
- **Zarr Data Access**: Always use `xarray.open_zarr()` instead of `zarr.open()` for inspecting or loading zarr datasets. Xarray provides labeled dimensions, coordinates, and better integration with the data pipeline.

## Remote run advice

@PROD_RUN.md

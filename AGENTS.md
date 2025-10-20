# Repository Guidelines

## Project Structure & Module Organization

- Source entry points: `main.py` (training), `cli.py` (CLI helpers).
- Core packages: `models/` (architectures), `graph/` (graph building), `training/` (loops, hooks), `utils/` (helpers).
- Data and outputs: `data/` (inputs, metadata), `outputs/` (artifacts, logs), `reports/` (figures/notes). Do not commit large files.
- Tests: `tests/` (pytest). Some legacy tests exist at repo root; prefer `tests/`.

## Build, Test, and Development Commands

- `make install` — sync runtime deps via `uv`.
- `make dev` — install with dev tooling (ruff, black, mypy, pytest).
- `make test` — run pytest (configured to `tests/`).
- `make lint` — ruff + mypy checks.
- `make format` — black + ruff --fix.
- `make train` — run a basic training experiment.
- `make demo` — longer demo with preprocessing and saved artifacts.
- Use `uv run <cmd>` to execute within the env (e.g., `uv run pytest -k windowing -q`).

## Coding Style & Naming Conventions

- Python 3.11+. Format with Black (88 chars). Lint with Ruff; type-check with mypy (py39 target).
- Indentation: 4 spaces; no tabs. Imports sorted (Ruff isort rules).
- Naming: modules `snake_case.py`, classes `PascalCase`, functions/vars `snake_case`, constants `UPPER_SNAKE_CASE`.
- Prefer type hints, dataclasses where appropriate, and small, pure functions.
- Tensor ops: prefer `einops` (`rearrange`, `reduce`, `repeat`) for matrix/tensor manipulation over manual `view/reshape/permute` for clarity.
- For Python types, prefer native list, dict, tuple instead of the typing library. Prefer the `x | None` notation over Optional

## Testing Guidelines

- Framework: pytest. Place tests under `tests/` as `test_*.py`.
- Cover new/changed logic with focused unit tests; avoid network/IO in unit tests.
- Use fixtures for sample data; keep tests deterministic (seed RNGs).
- Quick examples: `uv run pytest -k graph`, `uv run pytest -q`.

## Commit & Pull Request Guidelines

- Commits: imperative mood, scoped, and small (e.g., `graph: add OD edge builder`). Reference issues when applicable.
- PRs must include: concise description, rationale, screenshots of plots (if any) saved to `outputs/`, and notes on breaking changes or new Make targets.
- CI gates locally: ensure `make lint` and `make test` pass before requesting review.

## Agent-Specific Instructions

- Prefer Make targets and `uv` over ad-hoc commands; update `pyproject.toml` if adding deps.
- Do not modify `data/` inputs or commit large `outputs/`. Place generated artifacts in `outputs/` and ignore by default.
- Keep patches minimal and consistent with existing style; avoid unrelated refactors. If heavy compute is required, add flags and default to lightweight settings.
- This project is a work in progress. It is ok to make breaking changes as the interface evolves


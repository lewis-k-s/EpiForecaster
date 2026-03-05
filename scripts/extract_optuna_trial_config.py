#!/usr/bin/env python3
"""Extract and reconstruct config for a specific Optuna trial.

Reconstructs the full config used for a trial by:
1. Loading the base config (from trial user_attrs or --base-config)
2. Applying the trial's hyperparameter overrides
3. Writing the final config to a YAML file

Usage:
  # Extract trial 127 from horizon_fix study (base config stored in journal)
  uv run python scripts/extract_optuna_trial_config.py \
    --journal outputs/optuna/horizon_fix.journal \
    --study horizon_fix \
    --trial 127 \
    --output outputs/optuna/horizon_fix/trial_127_config.yaml

  # Override base config (for older sweeps without stored base)
  uv run python scripts/extract_optuna_trial_config.py \
    --journal outputs/optuna/horizon_fix.journal \
    --study horizon_fix \
    --trial 127 \
    --base-config configs/production_only/train_epifor_mn5_full.yaml \
    --output outputs/optuna/horizon_fix/trial_127_config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import importlib  # noqa: E402
from typing import Any  # noqa: E402

import click  # noqa: E402
import yaml  # noqa: E402

from models.configs import EpiForecasterConfig  # noqa: E402


def _overrides_to_dotlist(overrides: dict[str, Any]) -> list[str]:
    """Convert dict of overrides to dotlist format for OmegaConf."""
    dotlist = []
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value_str = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            value_str = str(value)
        else:
            value_str = str(value)
        dotlist.append(f"{key}={value_str}")
    return dotlist


@click.command()
@click.option(
    "--journal",
    "journal_file",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to Optuna journal file (.journal)",
)
@click.option("--study", "study_name", type=str, required=True, help="Study name")
@click.option("--trial", "trial_number", type=int, required=True, help="Trial number")
@click.option(
    "--base-config",
    "base_config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional base config path (overrides stored base if provided)",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for reconstructed config YAML",
)
def main(
    *,
    journal_file: Path,
    study_name: str,
    trial_number: int,
    base_config_path: Path | None,
    output_path: Path,
) -> None:
    """Extract and reconstruct config for a specific Optuna trial."""
    try:
        optuna = importlib.import_module("optuna")
        storages = importlib.import_module("optuna.storages")
        JournalStorage = getattr(storages, "JournalStorage")
        journal_module = importlib.import_module("optuna.storages.journal")
        JournalFileBackend = getattr(journal_module, "JournalFileBackend")
    except Exception as exc:
        raise click.ClickException(
            f"optuna is not available in this environment. Import error: {exc}"
        ) from exc

    storage = JournalStorage(JournalFileBackend(str(journal_file)))
    study = optuna.load_study(study_name=study_name, storage=storage)

    trial = None
    for t in study.trials:
        if t.number == trial_number:
            trial = t
            break

    if trial is None:
        raise click.ClickException(
            f"Trial {trial_number} not found in study '{study_name}'. "
            f"Available trials: {[t.number for t in study.trials]}"
        )

    if trial.state != optuna.trial.TrialState.COMPLETE:
        raise click.ClickException(
            f"Trial {trial_number} is not complete (state={trial.state.name}). "
            "Only completed trials can be extracted."
        )

    user_attrs = trial.user_attrs
    stored_base_config = user_attrs.get("base_config_path")

    if base_config_path is not None:
        resolved_base = base_config_path.resolve()
        if stored_base_config is not None:
            stored_path = Path(stored_base_config)
            if stored_path.resolve() != resolved_base:
                click.echo(
                    f"WARNING: Base config override provided (--base-config) "
                    f"differs from stored base in trial user_attrs:\n"
                    f"  Stored:   {stored_base_config}\n"
                    f"  Override: {base_config_path}\n"
                    f"Using override: {base_config_path}"
                )
        base_to_use = base_config_path
    elif stored_base_config is not None:
        base_to_use = Path(stored_base_config)
        if not base_to_use.exists():
            raise click.ClickException(
                f"Base config stored in trial user_attrs does not exist: {stored_base_config}\n"
                f"Provide --base-config to override."
            )
    else:
        raise click.ClickException(
            "No base config found in trial user_attrs and --base-config not provided.\n"
            "Please provide --base-config to specify the base config used for this study."
        )

    if not base_to_use.exists():
        raise click.ClickException(f"Base config file not found: {base_to_use}")

    user_attrs = trial.user_attrs
    overrides = user_attrs.get("overrides", {})

    if not overrides:
        raise click.ClickException(
            f"Trial {trial_number} has no 'overrides' in user_attrs. "
            "This may be an old trial format. Cannot reconstruct config."
        )

    override_list = _overrides_to_dotlist(overrides)

    cfg = EpiForecasterConfig.load(
        str(base_to_use),
        overrides=override_list,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = cfg.to_dict()
    with open(output_path, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Trial {trial_number} (value={trial.value:.6f})")
    click.echo(f"Base config: {base_to_use}")
    click.echo(f"Overrides: {len(overrides)} parameters")
    click.echo(f"Output: {output_path}")


if __name__ == "__main__":
    main()

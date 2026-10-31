"""Declarative ablation-campaign worker coordinated by Optuna JournalStorage.

The checked-in campaign is the scientific design authority. Optuna only owns
exactly-once queue coordination; it neither samples nor defines conditions.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import signal
import os
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import click
import optuna
import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.distributions import CategoricalDistribution
from optuna.trial import create_trial

from models.configs import EpiForecasterConfig
from scripts.hpo.worker_identity import compute_worker_seed as _compute_worker_seed
from training.epiforecaster_trainer import EpiForecasterTrainer
from utils.logging import setup_logging

logger = logging.getLogger(__name__)

QUEUE_INITIALIZED_ATTR = "ablation_queue_initialized"
CAMPAIGN_SCHEMA_VERSION = 1
TERMINAL_TRIAL_STATES = {
    optuna.trial.TrialState.COMPLETE,
    optuna.trial.TrialState.FAIL,
    optuna.trial.TrialState.PRUNED,
}

CLOSEOUT_EXIT_CODE = 75
CAMPAIGN_STAGES = frozenset({"ablation", "ablation_smoke"})
STALE_RECOVERED_ATTR = "ablation_stale_recovered_trials"
# Attempts killed with their allocation (wall time, node loss) are not model
# failures. They stop consuming nothing only up to this bound per identity, so
# a trial that always kills its worker still ends in a retry-budget error.
STALE_RECOVERY_BUDGET = 3


@dataclass(frozen=True)
class CampaignCondition:
    name: str
    hypothesis: str
    overrides: dict[str, Any]
    allowed_difference_paths: tuple[str, ...]


@dataclass(frozen=True)
class EqualBudgetPolicy:
    """Resolved configuration paths that must be equal for every condition."""

    paths: tuple[str, ...]


@dataclass(frozen=True)
class AblationCampaign:
    """Validated immutable scientific design compiled from campaign YAML."""

    path: Path
    sha256: str
    campaign_id: str
    stage: str
    hypothesis: str
    base_config: Path
    baseline_condition: str
    fixed_overrides: dict[str, Any]
    smoke_overrides: dict[str, Any]
    conditions: tuple[CampaignCondition, ...]
    paired_model_seeds: tuple[int, ...]
    split_manifest: Path | None
    fold_indices: tuple[int, ...]
    smoke_fold_indices: tuple[int, ...]
    primary_metric: str
    primary_direction: str
    secondary_metrics: tuple[str, ...]
    crossval_num_folds: int | None
    safety_metrics: tuple[str, ...]
    diagnostic_metrics: tuple[str, ...]
    equal_budget_policy: EqualBudgetPolicy
    completeness_requirements: dict[str, Any]
    provenance_requirements: dict[str, Any]
    allowed_reuse_code_identities: tuple[str, ...] = ()

    @property
    def has_folds(self) -> bool:
        return bool(self.fold_indices)

    @property
    def conditions_by_name(self) -> dict[str, CampaignCondition]:
        return {condition.name: condition for condition in self.conditions}

    def selected_fold_indices(self, *, smoketest: bool) -> tuple[int, ...]:
        """Return all production folds or the declaration-owned smoke subset."""
        if smoketest and self.smoke_fold_indices:
            return self.smoke_fold_indices
        return self.fold_indices


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return tuple(value)


def _dot_paths(value: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> set[str]:
    paths: set[str] = set()
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("override mappings require non-empty string keys")
        path = (*prefix, key)
        if isinstance(item, Mapping):
            paths.update(_dot_paths(item, path))
        else:
            paths.add(".".join(path))
    return paths


def _deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = json.loads(json.dumps(base))
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        elif isinstance(value, Mapping):
            merged[key] = _deep_merge({}, value)
        else:
            merged[key] = value
    return merged


def _normalize_overrides(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Accept nested mappings and explicit dot-path leaves as one structured form."""
    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key:
            raise ValueError("override mappings require non-empty string keys")
        parts = key.split(".")
        if any(not part for part in parts):
            raise ValueError(f"invalid dot-path override: {key!r}")
        cursor = normalized
        for part in parts[:-1]:
            existing = cursor.get(part)
            if existing is not None and not isinstance(existing, dict):
                raise ValueError(f"conflicting override path: {key!r}")
            cursor = cursor.setdefault(part, {})
        leaf = parts[-1]
        normalized_value = (
            _normalize_overrides(value) if isinstance(value, Mapping) else value
        )
        if leaf in cursor and cursor[leaf] != normalized_value:
            raise ValueError(f"conflicting override path: {key!r}")
        cursor[leaf] = normalized_value
    return normalized


def _promote_model_params(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror EpiForecasterConfig's ``model.params`` promotion for overrides."""
    normalized = _normalize_overrides(mapping)
    model = normalized.get("model")
    if not isinstance(model, Mapping) or "params" not in model:
        return normalized

    params = model["params"]
    if not isinstance(params, Mapping):
        raise ValueError("model.params override must be a mapping")
    promoted = {key: value for key, value in model.items() if key != "params"}
    conflicts = set(params).intersection(promoted)
    if any(params[key] != promoted[key] for key in conflicts):
        names = ", ".join(f"model.{key}" for key in sorted(conflicts))
        raise ValueError(f"conflicting model.params override paths: {names}")
    normalized["model"] = _deep_merge(params, promoted)
    return normalized


def _flatten(value: Any, prefix: tuple[str, ...] = ()) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {".".join(prefix): value}
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        flattened.update(_flatten(item, (*prefix, str(key))))
    return flattened


def _resolve_source_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else Path.cwd() / path


def _manifest_num_folds(path: Path) -> int:
    """Read the persisted manifest's explicit fold cardinality."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot parse cross-validation split manifest {path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(
            f"cross_validation.split_manifest must contain an object: {path}"
        )
    num_folds = manifest.get("num_folds")
    if not isinstance(num_folds, int) or isinstance(num_folds, bool) or num_folds <= 0:
        raise ValueError(
            f"cross_validation.split_manifest has invalid num_folds: {path}"
        )
    return num_folds


def _hydrate_effective_config(
    *,
    base_config: Path,
    overrides: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Return the explicit-default, model.params-promoted configuration identity."""
    try:
        config = EpiForecasterConfig.load(
            str(base_config),
            overrides=_as_cli_overrides(overrides),
        )
    except Exception as exc:
        raise ValueError(f"{label}: cannot hydrate EpiForecasterConfig: {exc}") from exc
    return config.to_dict()


def _resolved_value(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"declared budget path does not resolve: {path}")
        value = value[part]
    return value


def _validate_conditions(
    *,
    base_config: Path,
    fixed_overrides: dict[str, Any],
    conditions: tuple[CampaignCondition, ...],
    baseline_condition: str,
    equal_budget_policy: EqualBudgetPolicy,
) -> None:
    by_name = {condition.name: condition for condition in conditions}
    if baseline_condition not in by_name:
        raise ValueError(
            f"baseline_condition {baseline_condition!r} is not a declared condition"
        )

    resolved_by_name = {
        condition.name: _hydrate_effective_config(
            base_config=base_config,
            overrides=_deep_merge(fixed_overrides, condition.overrides),
            label=f"condition {condition.name!r}",
        )
        for condition in conditions
    }
    baseline = resolved_by_name[baseline_condition]
    baseline_flat = _flatten(baseline)
    semantic_identities: dict[str, str] = {}
    for condition in conditions:
        resolved = resolved_by_name[condition.name]
        resolved_flat = _flatten(resolved)
        changed_paths = {
            path
            for path in baseline_flat.keys() | resolved_flat.keys()
            if baseline_flat.get(path) != resolved_flat.get(path)
        }
        undeclared = changed_paths - set(condition.allowed_difference_paths)
        if undeclared:
            raise ValueError(
                f"condition {condition.name!r} has undeclared config drift: "
                f"{', '.join(sorted(undeclared))}"
            )
        identity = _canonical_json(resolved)
        if identity in semantic_identities:
            raise ValueError(
                f"conditions {semantic_identities[identity]!r} and "
                f"{condition.name!r} have the same semantic identity"
            )
        semantic_identities[identity] = condition.name

    baseline_budget = {
        path: _resolved_value(baseline, path) for path in equal_budget_policy.paths
    }
    for condition_name, resolved in resolved_by_name.items():
        for path, baseline_value in baseline_budget.items():
            value = _resolved_value(resolved, path)
            if value != baseline_value:
                raise ValueError(
                    f"condition {condition_name!r} changes equal budget path "
                    f"{path}: {baseline_value!r} -> {value!r}"
                )


def load_campaign(path: Path) -> AblationCampaign:
    """Load and validate a versioned campaign declaration before execution."""
    path = path.resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read campaign {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Cannot parse campaign {path}: {exc}") from exc
    root = _mapping(raw, field="campaign")
    required = {
        "schema_version",
        "campaign_id",
        "stage",
        "hypothesis",
        "base_config",
        "baseline_condition",
        "fixed_overrides",
        "conditions",
        "paired_model_seeds",
        "metrics",
        "equal_budget_policy",
        "completeness_requirements",
        "provenance_requirements",
    }
    missing = sorted(required - root.keys())
    if missing:
        raise ValueError(f"campaign is missing required fields: {', '.join(missing)}")
    if root["schema_version"] != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported campaign schema_version {root['schema_version']!r}; "
            f"expected {CAMPAIGN_SCHEMA_VERSION}"
        )
    campaign_id = _string(root["campaign_id"], field="campaign_id")
    if Path(campaign_id).name != campaign_id or campaign_id in {".", ".."}:
        raise ValueError("campaign_id must be one safe path component")
    base_config = _resolve_source_path(
        _string(root["base_config"], field="base_config")
    )
    if not base_config.is_file():
        raise ValueError(f"Cannot read base_config {base_config}")
    fixed_overrides = _promote_model_params(
        _mapping(root["fixed_overrides"], field="fixed_overrides")
    )
    smoke_overrides = _promote_model_params(
        _mapping(root.get("smoke_overrides", {}), field="smoke_overrides")
    )
    raw_conditions = root["conditions"]
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("conditions must be a non-empty list")
    conditions: list[CampaignCondition] = []
    names: set[str] = set()
    for index, raw_condition in enumerate(raw_conditions):
        item = _mapping(raw_condition, field=f"conditions[{index}]")
        for field in ("name", "hypothesis", "overrides", "allowed_difference_paths"):
            if field not in item:
                raise ValueError(f"conditions[{index}] is missing {field}")
        name = _string(item["name"], field=f"conditions[{index}].name")
        if name in names:
            raise ValueError(f"duplicate condition name: {name}")
        names.add(name)
        conditions.append(
            CampaignCondition(
                name=name,
                hypothesis=_string(
                    item["hypothesis"], field=f"conditions[{index}].hypothesis"
                ),
                overrides=_promote_model_params(
                    _mapping(item["overrides"], field=f"conditions[{index}].overrides")
                ),
                allowed_difference_paths=_string_list(
                    item["allowed_difference_paths"],
                    field=f"conditions[{index}].allowed_difference_paths",
                ),
            )
        )
    seeds_raw = root["paired_model_seeds"]
    if (
        not isinstance(seeds_raw, list)
        or not seeds_raw
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds_raw
        )
    ):
        raise ValueError("paired_model_seeds must be a non-empty list of integers")
    seeds = tuple(seeds_raw)
    if len(set(seeds)) != len(seeds):
        raise ValueError("paired_model_seeds must be unique")
    split_manifest: Path | None = None
    smoke_fold_indices: tuple[int, ...] = ()
    fold_indices: tuple[int, ...] = ()
    crossval_num_folds: int | None = None
    if root.get("cross_validation") is not None:
        cv = _mapping(root["cross_validation"], field="cross_validation")
        folds_raw = cv.get("fold_indices", [])
        if not isinstance(folds_raw, list) or any(
            not isinstance(fold, int) or isinstance(fold, bool) or fold < 0
            for fold in folds_raw
        ):
            raise ValueError(
                "cross_validation.fold_indices must be non-negative integers"
            )
        fold_indices = tuple(folds_raw)
        if len(set(fold_indices)) != len(fold_indices):
            raise ValueError("cross_validation.fold_indices must be unique")
        smoke_folds_raw = cv.get("smoke_fold_indices", [])
        if not isinstance(smoke_folds_raw, list) or any(
            not isinstance(fold, int) or isinstance(fold, bool) or fold < 0
            for fold in smoke_folds_raw
        ):
            raise ValueError(
                "cross_validation.smoke_fold_indices must be non-negative integers"
            )
        smoke_fold_indices = tuple(smoke_folds_raw)
        if len(set(smoke_fold_indices)) != len(smoke_fold_indices):
            raise ValueError("cross_validation.smoke_fold_indices must be unique")
        if set(smoke_fold_indices).difference(fold_indices):
            raise ValueError(
                "cross_validation.smoke_fold_indices must be a subset of fold_indices"
            )
        manifest_raw = cv.get("split_manifest")
        if fold_indices and not isinstance(manifest_raw, str):
            raise ValueError(
                "cross_validation.split_manifest is required when folds are configured"
            )
        if manifest_raw is not None:
            split_manifest = _resolve_source_path(
                _string(manifest_raw, field="cross_validation.split_manifest")
            )
            if not split_manifest.is_file():
                raise ValueError(
                    f"cross_validation.split_manifest does not exist: {split_manifest}"
                )
            crossval_num_folds = _manifest_num_folds(split_manifest)
            invalid_folds = [
                fold for fold in fold_indices if fold >= crossval_num_folds
            ]
            if invalid_folds:
                raise ValueError(
                    "cross_validation.fold_indices exceed manifest num_folds "
                    f"{crossval_num_folds}: {invalid_folds}"
                )
    metrics = _mapping(root["metrics"], field="metrics")
    primary = _mapping(metrics.get("primary"), field="metrics.primary")
    completeness_requirements = _mapping(
        root["completeness_requirements"], field="completeness_requirements"
    )
    provenance_requirements = _mapping(
        root["provenance_requirements"], field="provenance_requirements"
    )
    required_completeness = {
        "require_exact_identity_matrix",
        "require_terminal_trials",
        "require_finite_primary_metric",
    }
    required_provenance = {"require_campaign_sha256", "require_output_identity"}
    for field, requirements in (
        ("completeness_requirements", required_completeness),
        ("provenance_requirements", required_provenance),
    ):
        mapping = (
            completeness_requirements
            if field == "completeness_requirements"
            else provenance_requirements
        )
        missing_requirements = requirements - mapping.keys()
        if missing_requirements or any(
            not isinstance(mapping.get(key), bool) for key in requirements
        ):
            requirement_names = ", ".join(sorted(missing_requirements or requirements))
            raise ValueError(
                f"{field} must declare boolean requirements: {requirement_names}"
            )
    require_full_manifest_folds = completeness_requirements.get(
        "require_full_manifest_folds", False
    )
    if not isinstance(require_full_manifest_folds, bool):
        raise ValueError(
            "completeness_requirements.require_full_manifest_folds must be boolean"
        )
    if require_full_manifest_folds:
        if crossval_num_folds is None:
            raise ValueError(
                "full manifest fold coverage requires cross_validation.split_manifest"
            )
        expected_folds = tuple(range(crossval_num_folds))
        if fold_indices != expected_folds:
            raise ValueError(
                "full manifest fold coverage requires ordered fold_indices "
                f"{list(expected_folds)}, got {list(fold_indices)}"
            )

    stage = _string(root["stage"], field="stage")
    if stage not in CAMPAIGN_STAGES:
        raise ValueError(
            f"stage must be one of {sorted(CAMPAIGN_STAGES)}, got {stage!r}"
        )
    if stage == "ablation":
        if fold_indices and not require_full_manifest_folds:
            raise ValueError(
                "stage 'ablation' with folds requires "
                "completeness_requirements.require_full_manifest_folds: true; "
                "use stage 'ablation_smoke' for bounded fold subsets"
            )
        if len(conditions) < 2:
            raise ValueError("stage 'ablation' requires at least two conditions")
        if len(seeds) < 2:
            raise ValueError("stage 'ablation' requires at least two paired seeds")
    allowed_reuse_code_identities: tuple[str, ...] = ()
    if root.get("reuse") is not None:
        reuse = _mapping(root["reuse"], field="reuse")
        if set(reuse) != {"allow_code_identities"}:
            raise ValueError("reuse must contain only allow_code_identities")
        allowed_reuse_code_identities = _string_list(
            reuse["allow_code_identities"], field="reuse.allow_code_identities"
        )
        if any(
            re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in allowed_reuse_code_identities
        ):
            raise ValueError(
                "reuse.allow_code_identities must be source bundle SHA-256 digests"
            )

    primary_name = _string(primary.get("name"), field="metrics.primary.name")
    primary_direction = _string(
        primary.get("direction"), field="metrics.primary.direction"
    )
    if primary_direction not in {"minimize", "maximize"}:
        raise ValueError("metrics.primary.direction must be 'minimize' or 'maximize'")
    raw_budget_policy = _mapping(
        root["equal_budget_policy"], field="equal_budget_policy"
    )
    if set(raw_budget_policy) != {"paths"}:
        raise ValueError("equal_budget_policy must contain only non-empty paths")
    budget_paths = _string_list(
        raw_budget_policy["paths"], field="equal_budget_policy.paths"
    )
    if not budget_paths or len(set(budget_paths)) != len(budget_paths):
        raise ValueError("equal_budget_policy.paths must be unique and non-empty")
    if any(
        any(not part for part in budget_path.split(".")) for budget_path in budget_paths
    ):
        raise ValueError("equal_budget_policy.paths must contain valid dot paths")
    campaign = AblationCampaign(
        path=path,
        sha256=hashlib.sha256(_canonical_json(root).encode("utf-8")).hexdigest(),
        campaign_id=campaign_id,
        stage=stage,
        hypothesis=_string(root["hypothesis"], field="hypothesis"),
        base_config=base_config,
        baseline_condition=_string(
            root["baseline_condition"], field="baseline_condition"
        ),
        smoke_fold_indices=smoke_fold_indices,
        fixed_overrides=fixed_overrides,
        smoke_overrides=smoke_overrides,
        conditions=tuple(conditions),
        paired_model_seeds=seeds,
        split_manifest=split_manifest,
        fold_indices=fold_indices,
        crossval_num_folds=crossval_num_folds,
        primary_metric=primary_name,
        primary_direction=primary_direction,
        secondary_metrics=_string_list(
            metrics.get("secondary", []), field="metrics.secondary"
        ),
        safety_metrics=_string_list(metrics.get("safety", []), field="metrics.safety"),
        diagnostic_metrics=_string_list(
            metrics.get("diagnostic", []), field="metrics.diagnostic"
        ),
        equal_budget_policy=EqualBudgetPolicy(paths=budget_paths),
        completeness_requirements=_mapping(
            root["completeness_requirements"], field="completeness_requirements"
        ),
        provenance_requirements=_mapping(
            root["provenance_requirements"], field="provenance_requirements"
        ),
        allowed_reuse_code_identities=allowed_reuse_code_identities,
    )
    _validate_conditions(
        base_config=base_config,
        fixed_overrides=fixed_overrides,
        conditions=campaign.conditions,
        baseline_condition=campaign.baseline_condition,
        equal_budget_policy=campaign.equal_budget_policy,
    )
    return campaign


def _as_cli_overrides(
    mapping: Mapping[str, Any], prefix: tuple[str, ...] = ()
) -> list[str]:
    overrides: list[str] = []
    for key, value in mapping.items():
        path = (*prefix, key)
        if isinstance(value, Mapping):
            overrides.extend(_as_cli_overrides(value, path))
        else:
            rendered = json.dumps(value, separators=(",", ":"))
            overrides.append(f"{'.'.join(path)}={rendered}")
    return overrides


def build_trial_queue(
    campaign: AblationCampaign, *, smoketest: bool = False
) -> list[dict[str, int | str]]:
    """Compile stable condition × fold × paired-model-seed identities."""
    folds: tuple[int | None, ...] = campaign.selected_fold_indices(
        smoketest=smoketest
    ) or (None,)
    queue: list[dict[str, int | str]] = []
    for condition in campaign.conditions:
        for fold in folds:
            for seed in campaign.paired_model_seeds:
                params: dict[str, int | str] = {
                    "condition": condition.name,
                    "seed": seed,
                }
                if fold is not None:
                    params["fold"] = fold
                queue.append(params)
    return queue


def _trial_identity(params: Mapping[str, object]) -> tuple[str, int, int | None] | None:
    condition, seed, fold = (
        params.get("condition"),
        params.get("seed"),
        params.get("fold"),
    )
    if not isinstance(condition, str) or not isinstance(seed, int):
        return None
    if fold is not None and not isinstance(fold, int):
        return None
    return condition, seed, fold


def trial_identity(
    trial: optuna.trial.FrozenTrial,
) -> tuple[str, int, int | None] | None:
    identity = _trial_identity(trial.params)
    if identity is not None:
        return identity
    fixed = trial.system_attrs.get("fixed_params", {})
    return _trial_identity(fixed) if isinstance(fixed, Mapping) else None


def expected_trial_identities(
    campaign: AblationCampaign, *, smoketest: bool = False
) -> set[tuple[str, int, int | None]]:
    return {
        identity
        for params in build_trial_queue(campaign, smoketest=smoketest)
        if (identity := _trial_identity(params)) is not None
    }


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _mark_running_attempt_failed(study: optuna.Study, trial_number: int) -> None:
    """Terminalize a stale attempt through Optuna's supported public API."""
    study.tell(trial_number, state=optuna.trial.TrialState.FAIL)


def _terminalize_stale_attempt(study: optuna.Study, trial_number: int) -> None:
    """Fail a RUNNING attempt left by a dead allocation and mark it as infrastructure.

    The mark keeps wall-time kills out of the model-failure retry budget.
    """
    recorded = {int(value) for value in study.user_attrs.get(STALE_RECOVERED_ATTR, [])}
    recorded.add(trial_number)
    study.set_user_attr(STALE_RECOVERED_ATTR, sorted(recorded))
    _mark_running_attempt_failed(study, trial_number)


def _budgeted_failures(
    study: optuna.Study, attempts: list[optuna.trial.FrozenTrial]
) -> int:
    """Count failed attempts that consume the retry budget of one identity."""
    stale = {int(value) for value in study.user_attrs.get(STALE_RECOVERED_ATTR, [])}
    failed = [
        trial
        for trial in attempts
        if trial.state
        in {optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED}
    ]
    if not stale:
        return len(failed)
    infrastructure = sum(trial.number in stale for trial in failed)
    return (
        len(failed)
        - infrastructure
        + max(0, infrastructure - STALE_RECOVERY_BUDGET)
    )


TRAINING_ONLY_EXCLUSIONS = (
    ("output",),
    ("training", "run_final_evaluation"),
    ("training", "plot_forecasts"),
)


def _deep_merge(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _without_evaluation_controls(config: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(config))
    for path in TRAINING_ONLY_EXCLUSIONS:
        cursor: Any = normalized
        for part in path[:-1]:
            if not isinstance(cursor, dict):
                break
            cursor = cursor.get(part)
        if isinstance(cursor, dict):
            cursor.pop(path[-1], None)
    return normalized


def hydrated_training_config(
    campaign: AblationCampaign, *, condition: str, fold: int, seed: int
) -> dict[str, Any]:
    """Resolve the canonical training config of one identity (receipt contract)."""
    merged = _deep_merge(
        campaign.fixed_overrides, campaign.conditions_by_name[condition].overrides
    )
    overrides = _as_cli_overrides(merged)
    overrides.extend(
        [
            f"training.seed={seed}",
            f"training.crossval_fold_index={fold}",
            "output.write_granular_eval=true",
        ]
    )
    return EpiForecasterConfig.load(
        str(campaign.base_config), overrides=overrides
    ).to_dict()


def hydrated_training_config_sha256(
    campaign: AblationCampaign, *, condition: str, fold: int, seed: int
) -> str:
    """Hash the training-relevant hydrated config; evaluation controls excluded."""
    identity = _without_evaluation_controls(
        hydrated_training_config(campaign, condition=condition, fold=fold, seed=seed)
    )
    return hashlib.sha256(_canonical_json(identity).encode()).hexdigest()


def _bare_sha256(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.removeprefix("sha256:") or None


@dataclass(frozen=True)
class ExecutionIdentity:
    """Code, container and input identities of the executing immutable run."""

    source_bundle_sha256: str
    container_digest: str
    dvc_md5_by_output: Mapping[str, str]


def load_execution_identity(manifest_path: Path) -> ExecutionIdentity:
    """Read the identities that prepare wrote into the run manifest."""
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    identity = _mapping(manifest.get("identity"), field="manifest.identity")
    targets = _mapping(
        _mapping(manifest.get("dvc"), field="manifest.dvc").get("targets"),
        field="manifest.dvc.targets",
    )
    source = _bare_sha256(identity.get("source_snapshot_sha256"))
    container = _bare_sha256(identity.get("container_digest"))
    if source is None or container is None:
        raise ValueError(
            "run manifest lacks source_snapshot_sha256 or container_digest identity"
        )
    return ExecutionIdentity(
        source_bundle_sha256=source,
        container_digest=container,
        dvc_md5_by_output={
            Path(str(target["output"])).as_posix(): str(target["md5"])
            for target in targets.values()
            if isinstance(target, Mapping) and "output" in target and "md5" in target
        },
    )


def reuse_rejection_reasons(
    record: Mapping[str, Any],
    *,
    campaign: AblationCampaign,
    execution: ExecutionIdentity,
) -> list[str]:
    """Return why a reuse record is not valid for this run; empty means accepted."""
    reasons: list[str] = []
    if record.get("status") != "reusable":
        reasons.append("audit_status_not_reusable")
    try:
        condition, seed, fold = (
            str(record["condition"]),
            int(record["seed"]),
            int(record["fold"]),
        )
        hydrated = hydrated_training_config(
            campaign, condition=condition, fold=fold, seed=seed
        )
    except (KeyError, TypeError, ValueError) as exc:
        return [*reasons, f"identity_unresolvable:{type(exc).__name__}"]
    expected_hash = hashlib.sha256(
        _canonical_json(_without_evaluation_controls(hydrated)).encode()
    ).hexdigest()
    if record.get("hydrated_training_config_sha256") != expected_hash:
        reasons.append("hydrated_training_config_mismatch")
    for field, config_key in (
        ("dataset_dvc_md5", "dataset_path"),
        ("region2vec_dvc_md5", "region2vec_path"),
    ):
        path = hydrated.get("data", {}).get(config_key)
        executing = (
            execution.dvc_md5_by_output.get(Path(str(path)).as_posix())
            if path is not None
            else None
        )
        if field == "region2vec_dvc_md5" and record.get(field) is None:
            continue
        if executing is None or record.get(field) != executing:
            reasons.append(f"{field}_mismatch")
    if _bare_sha256(record.get("container_digest")) != execution.container_digest:
        reasons.append("container_digest_mismatch")
    code = _bare_sha256(record.get("source_bundle_sha256"))
    if code != execution.source_bundle_sha256 and (
        code is None or code not in campaign.allowed_reuse_code_identities
    ):
        reasons.append("source_bundle_not_allowed")
    return reasons


def _write_reuse_acceptance(
    path: Path,
    *,
    campaign: AblationCampaign,
    execution: ExecutionIdentity,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> None:
    """Persist the acceptance decision; identical for every worker rank."""
    document = {
        "schema_version": 1,
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign.sha256,
        "execution_source_bundle_sha256": execution.source_bundle_sha256,
        "execution_container_digest": execution.container_digest,
        "allowed_code_identities": list(campaign.allowed_reuse_code_identities),
        "accepted": accepted,
        "rejected": rejected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_reuse_identities(
    receipt_path: Path | None,
    campaign: AblationCampaign,
    execution: ExecutionIdentity | None = None,
    acceptance_path: Path | None = None,
) -> set[tuple[str, int, int | None]]:
    """Accept only historical identities that match this run's code and inputs.

    Records that fail any identity check are logged, written to the acceptance
    receipt, and left out so the campaign trains them again.
    """
    if receipt_path is None:
        return set()
    if execution is None:
        raise ValueError("checkpoint reuse requires the executing run identity")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("campaign_id") != campaign.campaign_id
        or receipt.get("campaign_sha256") != campaign.sha256
    ):
        raise ValueError("checkpoint reuse receipt does not match campaign identity")
    expected = expected_trial_identities(campaign)
    identities: set[tuple[str, int, int | None]] = set()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in receipt.get("records", []):
        identity = (str(item["condition"]), int(item["seed"]), int(item["fold"]))
        if identity not in expected:
            raise ValueError("checkpoint reuse receipt contains undeclared identities")
        reasons = reuse_rejection_reasons(item, campaign=campaign, execution=execution)
        entry = {
            "condition": identity[0],
            "seed": identity[1],
            "fold": identity[2],
        }
        if reasons:
            rejected.append({**entry, "reasons": reasons})
        else:
            identities.add(identity)
            accepted.append(entry)
    if rejected:
        logger.warning(
            "Rejected %d of %d checkpoint reuse records; they will be retrained: %s",
            len(rejected),
            len(rejected) + len(accepted),
            json.dumps(rejected, sort_keys=True),
        )
    logger.info("Accepted %d checkpoint reuse records", len(accepted))
    if acceptance_path is not None:
        _write_reuse_acceptance(
            acceptance_path,
            campaign=campaign,
            execution=execution,
            accepted=accepted,
            rejected=rejected,
        )
    return identities


def _record_reused_trials(
    study: optuna.Study,
    campaign: AblationCampaign,
    identities: set[tuple[str, int, int | None]],
) -> None:
    """Mark receipt-backed identities complete without staging checkpoints."""
    existing = {trial_identity(trial) for trial in study.trials}
    for condition, seed, fold in sorted(identities):
        identity = (condition, seed, fold)
        if identity in existing:
            continue
        params: dict[str, int | str] = {"condition": condition, "seed": seed}
        distributions: dict[str, CategoricalDistribution] = {
            "condition": CategoricalDistribution(
                [item.name for item in campaign.conditions]
            ),
            "seed": CategoricalDistribution(list(campaign.paired_model_seeds)),
        }
        if fold is not None:
            params["fold"] = fold
            distributions["fold"] = CategoricalDistribution(list(campaign.fold_indices))
        study.add_trial(
            create_trial(
                params=params,
                distributions=distributions,
                value=0.0,
                user_attrs={
                    "campaign_sha256": campaign.sha256,
                    "campaign_id": campaign.campaign_id,
                    "campaign_condition": condition,
                    "campaign_seed": seed,
                    "campaign_fold": fold,
                    "optuna_value_purpose": "provenance_verified_checkpoint_reuse",
                    "best_val_selection_metric": "mean_target_wis",
                    "best_val_selection_value": 0.0,
                },
            )
        )


def initialize_study_queue(
    *,
    study: optuna.Study,
    campaign: AblationCampaign,
    lock_path: Path,
    smoketest: bool = False,
    requeue_running: bool = False,
    reuse_identities: set[tuple[str, int, int | None]] | None = None,
    recover_stale_running: bool = False,
    active_allocation_id: str | None = None,
) -> None:
    """Fill uncompleted identities, optionally replacing stale running attempts."""
    expected_params = build_trial_queue(campaign, smoketest=smoketest)
    expected = expected_trial_identities(campaign, smoketest=smoketest)
    with _file_lock(lock_path):
        recorded_hash = study.user_attrs.get("campaign_sha256")
        if recorded_hash is not None and recorded_hash != campaign.sha256:
            raise ValueError(
                "Study campaign_sha256 does not match the requested campaign declaration"
            )
        if recover_stale_running:
            if active_allocation_id is None:
                raise ValueError("stale-trial recovery requires an allocation id")
            for trial in tuple(study.trials):
                allocation_id = trial.user_attrs.get("slurm_allocation_id")
                if (
                    trial_identity(trial) in expected
                    and trial.state == optuna.trial.TrialState.RUNNING
                    and allocation_id != active_allocation_id
                ):
                    _terminalize_stale_attempt(study, trial.number)
        elif requeue_running:
            for trial in tuple(study.trials):
                if (
                    trial_identity(trial) in expected
                    and trial.state == optuna.trial.TrialState.RUNNING
                ):
                    _terminalize_stale_attempt(study, trial.number)
        if reuse_identities:
            _record_reused_trials(study, campaign, reuse_identities)
        attempts_by_identity: dict[
            tuple[str, int, int | None], list[optuna.trial.FrozenTrial]
        ] = {}
        for trial in study.trials:
            identity = trial_identity(trial)
            if identity in expected:
                attempts_by_identity.setdefault(identity, []).append(trial)

        replacements = []
        for params in expected_params:
            identity = _trial_identity(params)
            assert identity is not None
            attempts = attempts_by_identity.get(identity, [])
            if any(
                attempt.state == optuna.trial.TrialState.COMPLETE
                for attempt in attempts
            ):
                continue
            if any(attempt.state not in TERMINAL_TRIAL_STATES for attempt in attempts):
                continue
            if _budgeted_failures(study, attempts) >= FAILED_ATTEMPT_BUDGET:
                continue
            replacements.append(params)
        for params in replacements:
            study.enqueue_trial(params)
        if replacements or not study.user_attrs.get(QUEUE_INITIALIZED_ATTR, False):
            study.set_user_attr(QUEUE_INITIALIZED_ATTR, True)
        study.set_user_attr("campaign_sha256", campaign.sha256)
        study.set_user_attr("campaign_id", campaign.campaign_id)


def is_study_complete(
    study: optuna.Study, campaign: AblationCampaign, *, smoketest: bool = False
) -> bool:
    """Require one successful attempt per identity and no live campaign attempts."""
    expected = expected_trial_identities(campaign, smoketest=smoketest)
    complete_counts: dict[tuple[str, int, int | None], int] = {}
    for trial in study.trials:
        identity = trial_identity(trial)
        if identity not in expected or trial.state not in TERMINAL_TRIAL_STATES:
            return False
        if trial.state == optuna.trial.TrialState.COMPLETE:
            complete_counts[identity] = complete_counts.get(identity, 0) + 1
    return set(complete_counts) == expected and all(
        count == 1 for count in complete_counts.values()
    )


def study_completion_status(
    study: optuna.Study, campaign: AblationCampaign, *, smoketest: bool = False
) -> str:
    """Classify the journal after all worker ranks have stopped."""
    expected = expected_trial_identities(campaign, smoketest=smoketest)
    attempts_by_identity: dict[
        tuple[str, int, int | None], list[optuna.trial.FrozenTrial]
    ] = {identity: [] for identity in expected}
    for trial in study.trials:
        identity = trial_identity(trial)
        if identity not in expected:
            raise ValueError(f"Journal has an unexpected trial identity: {identity}")
        attempts_by_identity[identity].append(trial)

    pending = False
    for identity, attempts in attempts_by_identity.items():
        complete = sum(
            trial.state == optuna.trial.TrialState.COMPLETE for trial in attempts
        )
        active = sum(trial.state not in TERMINAL_TRIAL_STATES for trial in attempts)
        failed = _budgeted_failures(study, attempts)
        if complete > 1 or active > 1 or (complete and active):
            raise ValueError(f"Journal has duplicate attempts for {identity}")
        if not complete:
            if failed >= FAILED_ATTEMPT_BUDGET and not active:
                raise ValueError(f"Identity {identity} exhausted its retry budget")
            pending = True
    return "pending" if pending else "complete"


def _has_waiting_trials(
    study: optuna.Study,
    expected: set[tuple[str, int, int | None]],
) -> bool:
    return any(
        trial.state == optuna.trial.TrialState.WAITING
        and trial_identity(trial) in expected
        for trial in study.trials
    )


@contextmanager
def closeout_signal_event() -> Iterator[Event]:
    """Expose Slurm's pre-time-limit signal as a stop-claiming request."""
    closeout_requested = Event()
    previous_handler = signal.getsignal(signal.SIGUSR1)

    def request_closeout(signum: int, frame: object) -> None:
        del signum, frame
        closeout_requested.set()
        logger.warning("Received Slurm pre-time-limit signal; stopping new claims")

    signal.signal(signal.SIGUSR1, request_closeout)
    try:
        yield closeout_requested
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)


def _identity_params(identity: tuple[str, int, int | None]) -> dict[str, int | str]:
    params: dict[str, int | str] = {
        "condition": identity[0],
        "seed": identity[1],
    }
    if identity[2] is not None:
        params["fold"] = identity[2]
    return params


def _reenqueue_incomplete_identities(
    study: optuna.Study,
    campaign: AblationCampaign,
    expected: set[tuple[str, int, int | None]],
    *,
    attempts_budget: int,
) -> bool:
    """Requeue incomplete identities on drain so a session never ends early.

    Returns whether any identity was re-enqueued, i.e. whether work remains.
    Identities whose failed attempts exhausted the budget stay failed so the
    finalizer can reject the campaign loudly instead of retrying forever.
    """
    del campaign
    attempts_by_identity: dict[
        tuple[str, int, int | None], list[optuna.trial.FrozenTrial]
    ] = {}
    for trial in study.trials:
        identity = trial_identity(trial)
        if identity in expected:
            attempts_by_identity.setdefault(identity, []).append(trial)
    requeued = False
    for identity in sorted(expected):
        attempts = attempts_by_identity.get(identity, [])
        if any(trial.state == optuna.trial.TrialState.COMPLETE for trial in attempts):
            continue
        if any(trial.state not in TERMINAL_TRIAL_STATES for trial in attempts):
            continue
        failed = _budgeted_failures(study, attempts)
        if failed >= attempts_budget:
            logger.warning(
                "Identity %s exhausted its %d-attempt budget; leaving it failed",
                identity,
                attempts_budget,
            )
            continue
        study.enqueue_trial(_identity_params(identity))
        requeued = True
    if requeued:
        logger.warning(
            "Queue drained with incomplete identities; re-enqueued them for retry"
        )
    return requeued


FAILED_ATTEMPT_BUDGET = 3


def _reenqueue_failed_identity(
    study: optuna.Study,
    identity: tuple[str, int, int | None],
    *,
    attempts_budget: int,
) -> None:
    """Requeue a transiently failed identity so the session does not drain early."""
    attempts = [trial for trial in study.trials if trial_identity(trial) == identity]
    failed = _budgeted_failures(study, attempts)
    has_complete = any(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
        if trial_identity(trial) == identity
    )
    if has_complete or failed >= attempts_budget:
        logger.warning(
            "Identity %s has %d failed attempts; leaving it failed",
            identity,
            failed,
        )
        return
    if any(
        trial_identity(trial) == identity and trial.state not in TERMINAL_TRIAL_STATES
        for trial in study.trials
    ):
        return
    study.enqueue_trial(_identity_params(identity))
    logger.warning(
        "Re-enqueued failed identity %s (attempt %d of %d)",
        identity,
        failed + 1,
        attempts_budget,
    )


def run_queued_trials(
    *,
    study: optuna.Study,
    campaign: AblationCampaign,
    lock_path: Path,
    run_root: Path,
    n_trials: int | None,
    timeout_s: int | None,
    smoketest: bool,
    closeout_requested: Event | None = None,
) -> bool:
    """Execute queued entries and report whether another allocation is required."""
    expected = expected_trial_identities(campaign, smoketest=smoketest)
    started = time.monotonic()
    completed = 0
    while n_trials is None or completed < n_trials:
        stop_claiming = (
            closeout_requested is not None and closeout_requested.is_set()
        ) or (timeout_s is not None and time.monotonic() - started >= timeout_s)
        if stop_claiming:
            with _file_lock(lock_path):
                return _has_waiting_trials(study, expected)
        with _file_lock(lock_path):
            waiting = [
                trial
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.WAITING
                and trial_identity(trial) in expected
            ]
            if not waiting:
                requeued = _reenqueue_incomplete_identities(
                    study, campaign, expected, attempts_budget=FAILED_ATTEMPT_BUDGET
                )
                if not requeued:
                    return False
                continue
            trial = study.ask()
            identity = _trial_identity(trial.system_attrs.get("fixed_params", {}))
            if identity not in expected:
                raise RuntimeError(
                    "JournalStorage returned an identity outside the campaign queue"
                )
            allocation_id = os.environ.get("SLURM_JOB_ID")
            if allocation_id is not None:
                trial.set_user_attr("slurm_allocation_id", allocation_id)
        try:
            value = objective(
                trial,
                campaign=campaign,
                run_root=run_root,
                smoketest=smoketest,
            )
        except optuna.TrialPruned:
            with _file_lock(lock_path):
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                _reenqueue_failed_identity(
                    study, identity, attempts_budget=FAILED_ATTEMPT_BUDGET
                )
        except Exception:
            logger.exception("Campaign trial %d failed", trial.number)
            with _file_lock(lock_path):
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                _reenqueue_failed_identity(
                    study, identity, attempts_budget=FAILED_ATTEMPT_BUDGET
                )
        else:
            with _file_lock(lock_path):
                study.tell(trial, value)
        completed += 1
    with _file_lock(lock_path):
        return _has_waiting_trials(study, expected)


def _set_campaign_trial_attrs(
    trial: optuna.Trial,
    campaign: AblationCampaign,
    condition: str,
    seed: int,
    fold: int | None,
) -> None:
    for key, value in {
        "campaign_sha256": campaign.sha256,
        "campaign_id": campaign.campaign_id,
        "campaign_condition": condition,
        "campaign_seed": seed,
        "campaign_fold": fold,
        "campaign_path": str(campaign.path),
        "optuna_value_purpose": "validation_selection_coordination",
    }.items():
        trial.set_user_attr(key, value)


def objective(
    trial: optuna.Trial,
    *,
    campaign: AblationCampaign,
    run_root: Path,
    smoketest: bool = False,
) -> float:
    condition_name = trial.suggest_categorical(
        "condition", [item.name for item in campaign.conditions]
    )
    seed = int(trial.suggest_categorical("seed", list(campaign.paired_model_seeds)))
    fold = (
        int(trial.suggest_categorical("fold", list(campaign.fold_indices)))
        if campaign.has_folds
        else None
    )
    condition = campaign.conditions_by_name[condition_name]
    _set_campaign_trial_attrs(trial, campaign, condition_name, seed, fold)
    overrides = _as_cli_overrides(campaign.fixed_overrides) + _as_cli_overrides(
        condition.overrides
    )
    overrides += [
        f"training.seed={seed}",
        "training.plot_forecasts=false",
        "output.write_granular_eval=true",
    ]
    if fold is not None:
        assert campaign.split_manifest is not None
        assert campaign.crossval_num_folds is not None
        overrides += [
            "training.crossval_enabled=true",
            f"training.crossval_num_folds={campaign.crossval_num_folds}",
            f"training.crossval_fold_index={fold}",
            f"training.crossval_manifest_path={campaign.split_manifest}",
        ]
    if smoketest:
        overrides += _as_cli_overrides(campaign.smoke_overrides)
    experiment_name = f"mn5_ablation__{campaign.campaign_id}__{condition_name}"
    tags = [
        "mn5",
        "ablation",
        f"campaign_{campaign.campaign_id}",
        f"condition_{condition_name}",
        f"seed_{seed}",
        f"campaign_sha256_{campaign.sha256[:48]}",
    ]
    if fold is not None:
        tags.append(f"fold_{fold}")
    overrides += [
        f"output.experiment_name={experiment_name}",
        f"output.wandb_group=ablation_{campaign.campaign_id}",
        f"output.wandb_tags={json.dumps(tags, separators=(',', ':'))}",
    ]
    if smoketest:
        smoke_training = campaign.smoke_overrides.get("training", {})
        if "max_batches" not in smoke_training:
            overrides.append("training.max_batches=2")
        if "epochs" not in smoke_training:
            overrides.append("training.epochs=1")
    cfg = EpiForecasterConfig.load(str(campaign.base_config), overrides=overrides)
    cfg.output.log_dir = str(run_root / campaign.campaign_id)
    model_id = (
        f"ablation_t{trial.number}_f{fold if fold is not None else 'none'}_s{seed}"
    )
    trainer = EpiForecasterTrainer(cfg, model_id=model_id)
    results = trainer.run()
    if trainer.experiment_dir is not None:
        output_identity = trainer.experiment_dir / "campaign_identity.json"
        output_identity.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "campaign_sha256": campaign.sha256,
                    "campaign_id": campaign.campaign_id,
                    "condition": condition_name,
                    "seed": seed,
                    "fold": fold,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        trial.set_user_attr("campaign_output_identity", str(output_identity))
    selection_value = float(results["best_val_selection_score"])
    trial.set_user_attr("best_val_selection_value", selection_value)
    trial.set_user_attr(
        "best_val_selection_metric", str(results["best_val_selection_metric"])
    )
    return selection_value


@click.command()
@click.option("--campaign", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--study-name", type=str, required=True)
@click.option("--journal-file", type=click.Path(path_type=Path), default=None)
@click.option(
    "--run-root",
    type=click.Path(path_type=Path),
    default=Path("outputs/training"),
    show_default=True,
)
@click.option("--n-trials", type=int, default=None)
@click.option("--timeout-s", type=int, default=None)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Worker coordination seed; not a design variable.",
)
@click.option("--smoketest", is_flag=True, default=False)
@click.option("--requeue-running", is_flag=True, default=False)
@click.option("--check-complete", is_flag=True, default=False)
@click.option(
    "--reuse-receipt", type=click.Path(exists=True, path_type=Path), default=None
)
@click.option("--recover-stale-running", is_flag=True, default=False)
@click.option(
    "--run-manifest",
    type=click.Path(path_type=Path),
    default=None,
    help="Run manifest.yaml; defaults to <run-root>/../../manifest.yaml.",
)
def main(
    campaign: Path,
    study_name: str,
    journal_file: Path | None,
    run_root: Path,
    n_trials: int | None,
    timeout_s: int | None,
    seed: int,
    smoketest: bool,
    requeue_running: bool,
    check_complete: bool,
    reuse_receipt: Path | None,
    recover_stale_running: bool,
    run_manifest: Path | None,
) -> None:
    if check_complete:
        declaration = load_campaign(campaign)
        if journal_file is None:
            journal_file = Path("outputs/ablation_studies") / f"{study_name}.journal"
        study = optuna.load_study(
            study_name=study_name,
            storage=JournalStorage(JournalFileBackend(str(journal_file))),
        )
        status = study_completion_status(study, declaration, smoketest=smoketest)
        if status == "pending":
            raise click.exceptions.Exit(CLOSEOUT_EXIT_CODE)
        return
    active_allocation_id = os.environ.get("SLURM_JOB_ID")
    if recover_stale_running and active_allocation_id is None:
        raise click.UsageError("--recover-stale-running requires SLURM_JOB_ID")
    setup_logging()
    declaration = load_campaign(campaign)
    run_root.mkdir(parents=True, exist_ok=True)
    if journal_file is None:
        journal_file = Path("outputs/ablation_studies") / f"{study_name}.journal"
    journal_file.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=JournalStorage(JournalFileBackend(str(journal_file))),
        sampler=optuna.samplers.RandomSampler(seed=_compute_worker_seed(seed)),
        direction="minimize",
        load_if_exists=True,
    )
    if study.direction != optuna.study.StudyDirection.MINIMIZE:
        raise ValueError(
            "Ablation Optuna study direction must minimize validation-selection "
            "coordination values"
        )
    queue_lock_path = journal_file.with_suffix(f"{journal_file.suffix}.queue.lock")
    execution: ExecutionIdentity | None = None
    acceptance_path: Path | None = None
    if reuse_receipt is not None:
        manifest_path = (
            run_manifest
            if run_manifest is not None
            else run_root.parent.parent / "manifest.yaml"
        )
        execution = load_execution_identity(manifest_path)
        acceptance_path = run_root / "reuse_acceptance.json"
    initialize_study_queue(
        study=study,
        campaign=declaration,
        lock_path=queue_lock_path,
        smoketest=smoketest,
        requeue_running=requeue_running,
        reuse_identities=load_reuse_identities(
            reuse_receipt, declaration, execution, acceptance_path
        ),
        recover_stale_running=recover_stale_running,
        active_allocation_id=active_allocation_id,
    )
    with closeout_signal_event() as closeout_requested:
        continuation_required = run_queued_trials(
            study=study,
            campaign=declaration,
            lock_path=queue_lock_path,
            run_root=run_root,
            n_trials=n_trials,
            timeout_s=timeout_s,
            smoketest=smoketest,
            closeout_requested=closeout_requested,
        )
    if continuation_required:
        logger.warning(
            "Ablation worker stopped before the queue drained; continuation is required"
        )
        raise click.exceptions.Exit(CLOSEOUT_EXIT_CODE)


if __name__ == "__main__":
    main()

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

from models.configs import EpiForecasterConfig
from scripts.hpo.worker_identity import compute_worker_seed as _compute_worker_seed
from training.native_trainer import NativeTrainer
from data.build_native_source_sets import build_for_config
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
        crossval_num_folds = cv.get("num_folds", 10)
        if type(crossval_num_folds) is not int or crossval_num_folds < 3:
            raise ValueError(
                "cross_validation.num_folds must be an integer of at least three"
            )
        if any(fold >= crossval_num_folds for fold in fold_indices):
            raise ValueError("cross_validation.fold_indices exceed num_folds")
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
            names = ", ".join(sorted(missing_requirements or requirements))
            raise ValueError(f"{field} must declare boolean requirements: {names}")

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
        stage=_string(root["stage"], field="stage"),
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


def initialize_study_queue(
    *,
    study: optuna.Study,
    campaign: AblationCampaign,
    lock_path: Path,
    smoketest: bool = False,
    requeue_running: bool = False,
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
        if requeue_running:
            for trial in tuple(study.trials):
                if (
                    trial_identity(trial) in expected
                    and trial.state == optuna.trial.TrialState.RUNNING
                ):
                    _mark_running_attempt_failed(study, trial.number)

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
                return False
            trial = study.ask()
            identity = _trial_identity(trial.system_attrs.get("fixed_params", {}))
            if identity not in expected:
                raise RuntimeError(
                    "JournalStorage returned an identity outside the campaign queue"
                )
        try:
            value = objective(
                trial,
                campaign=campaign,
                run_root=run_root,
                smoketest=smoketest,
            )
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except Exception:
            logger.exception("Campaign trial %d failed", trial.number)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
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
        assert campaign.crossval_num_folds is not None
        overrides += [
            f"native.num_folds={campaign.crossval_num_folds}",
            f"native.fold_index={fold}",
        ]
    if smoketest:
        overrides += _as_cli_overrides(campaign.smoke_overrides)
    experiment_name = f"mn5_ablation__{campaign.campaign_id}__{condition_name}"
    overrides += [
        f"output.experiment_name={experiment_name}",
        "output.wandb_mode=disabled",
    ]
    if smoketest:
        overrides += ["training.max_batches=2", "training.epochs=1"]
    cfg = EpiForecasterConfig.load(str(campaign.base_config), overrides=overrides)
    cfg.output.log_dir = str(run_root / campaign.campaign_id)
    model_id = (
        f"ablation_t{trial.number}_f{fold if fold is not None else 'none'}_s{seed}"
    )
    cfg.native.fold_manifest = str(
        Path(cfg.output.log_dir)
        / experiment_name
        / model_id
        / "inputs/native_source_sets/manifest.json"
    )
    build_for_config(cfg, Path(cfg.native.fold_manifest))
    trainer = NativeTrainer(cfg, model_id=model_id)
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
) -> None:
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
    initialize_study_queue(
        study=study,
        campaign=declaration,
        lock_path=queue_lock_path,
        smoketest=smoketest,
        requeue_running=requeue_running,
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

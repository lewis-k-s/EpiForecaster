from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.optuna_epiforecaster_worker import (
    _parse_batch_grad_combo,
    suggest_epiforecaster_params,
)


class _StubTrial:
    def __init__(self, categorical_values: dict[str, object] | None = None):
        self._categorical_values = categorical_values or {}
        self.params: dict[str, object] = {}

    def suggest_float(
        self, name: str, low: float, high: float, log: bool = False
    ) -> float:
        del high, log
        self.params[name] = low
        return low

    def suggest_int(self, name: str, low: int, high: int) -> int:
        del high
        self.params[name] = low
        return low

    def suggest_categorical(self, name: str, choices: tuple[object, ...]) -> object:
        value = self._categorical_values.get(name, choices[0])
        self.params[name] = value
        return value


def _base_cfg_stub() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            type=SimpleNamespace(mobility=False, regions=False),
            temporal_covariates_dim=0,
            include_day_of_week=True,
            include_holidays=True,
            input_window_length=60,
            forecast_horizon=28,
        ),
        training=SimpleNamespace(loss=SimpleNamespace(name="mse")),
    )


def test_batch_grad_combo_json_string_decodes_and_parses() -> None:
    assert _parse_batch_grad_combo("[32, 4]") == (32, 4)


@pytest.mark.parametrize("combo", [(64, 2), [64, 2]])
def test_batch_grad_combo_native_sequence_parses(
    combo: tuple[int, int] | list[int],
) -> None:
    assert _parse_batch_grad_combo(combo) == (64, 2)


@pytest.mark.parametrize("combo", ["[32, 4, 8]", "[]"])
def test_batch_grad_combo_invalid_shape_raises(combo: str) -> None:
    with pytest.raises(ValueError, match="training.batch_grad_combo"):
        _parse_batch_grad_combo(combo)


@pytest.mark.parametrize("combo", ['["32", 4]', ["32", 4], [0, 4], [-1, 2]])
def test_batch_grad_combo_invalid_types_raises(combo: object) -> None:
    with pytest.raises(ValueError, match="training.batch_grad_combo"):
        _parse_batch_grad_combo(combo)


def test_suggest_epiforecaster_params_sets_batch_and_grad_from_encoded_combo() -> None:
    trial = _StubTrial(categorical_values={"training.batch_grad_combo": "[32, 4]"})
    overrides = suggest_epiforecaster_params(
        trial=trial,
        base_cfg=_base_cfg_stub(),
    )
    assert overrides["training.batch_size"] == 32
    assert overrides["training.gradient_accumulation_steps"] == 4
    assert isinstance(overrides["training.batch_size"], int)
    assert isinstance(overrides["training.gradient_accumulation_steps"], int)

"""Precision and logging configuration boundaries."""

import pytest

from models.configs import TrainingParams


def test_training_precision_defaults_and_valid_override() -> None:
    defaults = TrainingParams()
    assert (
        defaults.parameter_dtype,
        defaults.mixed_precision_dtype,
        defaults.enable_mixed_precision,
        defaults.optimizer_eps,
    ) == ("float32", "bfloat16", True, None)

    configured = TrainingParams(enable_mixed_precision=False, optimizer_eps=1e-6)
    assert configured.enable_mixed_precision is False
    assert configured.optimizer_eps == 1e-6


def test_training_precision_rejects_unsupported_values() -> None:
    invalid = [
        ({"mixed_precision_dtype": "float16"}, "FP16 is no longer supported"),
        ({"mixed_precision_dtype": "FLOAT16"}, "FP16 is no longer supported"),
        ({"mixed_precision_dtype": "float64"}, "Unsupported mixed_precision_dtype"),
        ({"parameter_dtype": "float16"}, "Unsupported parameter_dtype"),
        ({"parameter_dtype": "bfloat16"}, "Unsupported parameter_dtype"),
        ({"optimizer_eps": 0.0}, "optimizer_eps must be positive"),
        ({"optimizer_eps": -1e-8}, "optimizer_eps must be positive"),
    ]
    for options, message in invalid:
        with pytest.raises(ValueError, match=message):
            TrainingParams(**options)


def test_training_logging_frequencies_reject_invalid_values() -> None:
    invalid = [
        ("wandb_gradient_histogram_frequency", -1, "must be non-negative"),
        ("wandb_gradient_histogram_max_params", 0, "must be >= 1"),
        ("model_diagnostics_frequency", -1, "must be non-negative"),
        ("pathway_gradient_frequency", -1, "must be non-negative"),
    ]
    for field, value, message in invalid:
        with pytest.raises(ValueError, match=message):
            TrainingParams(**{field: value})

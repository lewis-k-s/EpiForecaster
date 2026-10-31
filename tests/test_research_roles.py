import copy
import json

import pandas as pd
import pytest
import yaml

from data.research_roles import (
    load_role_manifest,
    research_time_ranges,
    validate_preprocessing_role,
)


@pytest.fixture
def manifest():
    return {
        "schema_version": 1,
        "source_audit_status": "complete",
        "roles": {
            "fit": {"label_start": "2020-05-01", "label_end": "2020-09-30"},
            "selection": {"label_start": "2020-10-29", "label_end": "2020-11-30"},
            "calibration": {"label_start": "2020-12-29", "label_end": "2021-01-31"},
            "confirmation": {"label_start": "2021-03-01", "label_end": "2021-05-09"},
        },
    }


def test_research_labels_and_gaps_are_disjoint(manifest):
    dates = pd.date_range("2020-03-01", "2021-05-09")
    ranges = research_time_ranges(dates.values, manifest, input_window=60, horizon=28)
    for role, (start, stop) in ranges.items():
        assert dates[start + 60] == pd.Timestamp(manifest["roles"][role]["label_start"])
        assert dates[stop - 1] == pd.Timestamp(manifest["roles"][role]["label_end"])
        # Every possible complete forecast stays within its role's label dates.
        for origin in range(start + 60, stop - 28 + 1):
            assert dates[origin + 27] <= pd.Timestamp(
                manifest["roles"][role]["label_end"]
            )


@pytest.mark.parametrize(
    "change", ["overlap", "gap", "history", "short", "missing_date"]
)
def test_research_rejects_invalid_support(manifest, change):
    dates = pd.date_range("2020-03-01", "2021-05-09")
    if change == "overlap":
        manifest["roles"]["selection"]["label_start"] = "2020-09-29"
    elif change == "gap":
        manifest["roles"]["selection"]["label_start"] = "2020-10-28"
    elif change == "history":
        manifest["roles"]["fit"]["label_start"] = "2020-03-02"
    elif change == "short":
        manifest["roles"]["selection"]["label_end"] = "2020-11-03"
    else:
        dates = dates.delete(1)
    with pytest.raises(ValueError):
        research_time_ranges(dates.values, manifest, input_window=60, horizon=28)


def test_research_preprocessing_must_exclude_later_labels(manifest):
    config = {
        "smoothing": {"parameter_fit_end_date": "2020-09-30"},
        "source_provenance": {"causal_preprocessing_contract": "loss-research-v1"},
    }
    validate_preprocessing_role(
        {"preprocessing_config_yaml": yaml.safe_dump(config)}, manifest
    )
    for cutoff in [None, "2021-05-09"]:
        bad = copy.deepcopy(config)
        bad["smoothing"]["parameter_fit_end_date"] = cutoff
        with pytest.raises(ValueError, match="cutoff"):
            validate_preprocessing_role(
                {"preprocessing_config_yaml": yaml.safe_dump(bad)}, manifest
            )


def test_role_manifest_hash_and_audit_gate(tmp_path, manifest):
    path = tmp_path / "roles.json"
    path.write_text(json.dumps(manifest))
    loaded, digest = load_role_manifest(path)
    assert loaded == manifest and len(digest) == 64
    manifest["source_audit_status"] = "pending"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source audit"):
        load_role_manifest(path)


@pytest.mark.parametrize("confirmation", [False, True])
@pytest.mark.parametrize("crossval", [False, True])
def test_dataset_factory_enforces_role_ranges_and_reuses_fit_scalers(
    tmp_path, monkeypatch, manifest, confirmation, crossval
):
    from types import SimpleNamespace

    import xarray as xr

    import data.dataset_factory as factory
    from models.configs import TrainingParams

    path = tmp_path / "roles.json"
    path.write_text(json.dumps(manifest))
    attrs = {
        "preprocessing_config_yaml": yaml.safe_dump(
            {
                "smoothing": {"parameter_fit_end_date": "2020-09-30"},
                "source_provenance": {
                    "causal_preprocessing_contract": "loss-research-v1"
                },
            }
        )
    }
    aligned = xr.Dataset(
        {"valid_targets": (("region_id",), [True, False, True])},
        coords={
            "date": pd.date_range("2020-03-01", "2021-05-09"),
            "region_id": ["a", "b", "c"],
        },
        attrs=attrs,
    )
    created = []

    class Dataset:
        @staticmethod
        def load_canonical_dataset(*args, **kwargs):
            return aligned

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for field in [
                "biomarker_preprocessor",
                "mobility_preprocessor",
                "preloaded_mobility",
                "mobility_mask",
                "shared_series",
            ]:
                setattr(self, field, kwargs.get(field, object()))
            created.append(self)

        def __len__(self):
            return 1

    monkeypatch.setattr(factory, "EpiDataset", Dataset)
    monkeypatch.setattr(
        factory, "split_nodes_by_ratio", lambda **kwargs: ([0], [1], [2])
    )
    training = TrainingParams(
        split_strategy="research",
        research_role_manifest_path=str(path),
        crossval_enabled=crossval,
        research_evaluation_role="confirmation" if confirmation else "calibration",
        research_confirmation_enabled=confirmation,
    )
    config = SimpleNamespace(
        training=training,
        data=SimpleNamespace(
            dataset_path="unused",
            run_id="real",
            run_id_chunk_size=1,
            use_valid_targets=True,
        ),
        model=SimpleNamespace(input_window_length=60, forecast_horizon=28),
    )
    result = factory._build_research_splits(config, None)
    ranges = research_time_ranges(
        aligned.date.values, manifest, input_window=60, horizon=28
    )
    assert result.train.kwargs["time_range"] == ranges["fit"]
    assert result.val.kwargs["time_range"] == ranges["selection"]
    role = "confirmation" if confirmation else "calibration"
    assert result.test.kwargs["time_range"] == ranges[role]
    assert len(created) == 3  # No hidden construction of the protected fourth role.
    for ds in [result.val, result.test]:
        assert ds.biomarker_preprocessor is result.train.biomarker_preprocessor
        assert ds.mobility_preprocessor is result.train.mobility_preprocessor
        assert ds.shared_series is result.train.shared_series
    expected_nodes = [2] if crossval else [0, 2]
    assert result.test.kwargs["target_nodes"] == expected_nodes
    assert result.test.kwargs["context_nodes"] == [0, 1, 2]


def test_confirmation_cannot_be_enabled_by_a_role_name_alone():
    from models.configs import TrainingParams

    with pytest.raises(ValueError, match="frozen-finalist"):
        TrainingParams(
            split_strategy="research",
            research_role_manifest_path="roles.json",
            research_evaluation_role="confirmation",
        )

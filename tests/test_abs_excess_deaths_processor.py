"""ABS all-cause excess mortality store tests."""

import numpy as np
import pandas as pd
import pytest

from data.preprocess.processors.abs_excess_deaths_processor import (
    build_abs_excess_deaths,
)


def _source() -> pd.DataFrame:
    dates = pd.date_range("2017-01-01", "2020-03-02")
    counts = np.where(dates.year < 2020, 2, 3)
    counts[dates == pd.Timestamp("2020-03-01")] = 1
    return pd.DataFrame(
        {
            "data": dates,
            "codi_abs": "001",
            "nom_abs": "Area 001",
            "sexe": "Home",
            "grup_edat": "55 a 59",
            "index_socioeconomic": "3",
            "casos": counts,
        }
    )


def test_signed_excess_uses_only_pre_pandemic_baseline() -> None:
    dataset = build_abs_excess_deaths(
        _source(),
        ["001", "002"],
        start=pd.Timestamp("2020-02-28"),
        end=pd.Timestamp("2020-03-02"),
    )
    assert dataset.sizes == {"date": 4, "abs_id": 2}
    assert dataset["expected"].sel(abs_id="001").values == pytest.approx([2] * 4)
    assert dataset["excess"].sel(abs_id="001").values == pytest.approx([1, 1, -1, 1])
    assert not dataset["mask"].sel(abs_id="002").any().item()
    assert dataset["mask"].sel(abs_id="001").all().item()
    assert dataset.attrs["unmatched_graph_ids"] == "002"


def test_duplicate_demographic_row_is_rejected() -> None:
    source = _source()
    with pytest.raises(ValueError, match="duplicate demographic keys"):
        build_abs_excess_deaths(
            pd.concat([source, source.iloc[[0]]], ignore_index=True),
            ["001"],
            start=pd.Timestamp("2020-02-28"),
            end=pd.Timestamp("2020-03-02"),
        )

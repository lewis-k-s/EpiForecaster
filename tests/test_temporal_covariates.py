import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from data.preprocess.config import (
    PreprocessingConfig,
    TemporalCovariatesConfig,
    TEMPORAL_COVARIATE_DIM,
)
from data.preprocess.processors.temporal_covariates_processor import (
    TemporalCovariatesProcessor,
)


class TestTemporalCovariatesConfig:
    def test_default_config(self):
        config = TemporalCovariatesConfig(holiday_calendar_file="/dev/null")
        assert config.include_day_of_week is True
        assert config.include_holidays is True

    def test_output_dim_both_enabled(self, tmp_path):
        holiday_file = tmp_path / "holidays.csv"
        holiday_file.write_text("date\n2020-01-01\n")
        config = TemporalCovariatesConfig(
            include_day_of_week=True,
            include_holidays=True,
            holiday_calendar_file=str(holiday_file),
        )
        assert config.output_dim == 3

    def test_output_dim_dow_only(self):
        config = TemporalCovariatesConfig(
            include_day_of_week=True,
            include_holidays=False,
            holiday_calendar_file=None,
        )
        assert config.output_dim == 2

    def test_output_dim_holidays_only(self, tmp_path):
        holiday_file = tmp_path / "holidays.csv"
        holiday_file.write_text("date\n2020-01-01\n")
        config = TemporalCovariatesConfig(
            include_day_of_week=False,
            include_holidays=True,
            holiday_calendar_file=str(holiday_file),
        )
        assert config.output_dim == 1

    def test_holiday_file_required_when_holidays_enabled(self):
        with pytest.raises(ValueError, match="holiday_calendar_file is required"):
            TemporalCovariatesConfig(
                include_holidays=True,
                holiday_calendar_file=None,
            )


class TestTemporalCovariatesProcessor:
    @pytest.fixture
    def sample_holiday_file(self, tmp_path: Path) -> Path:
        holiday_file = tmp_path / "holidays.csv"
        holiday_file.write_text(
            "date\n2020-01-01\n2020-12-25\n2020-09-11\n2021-01-01\n"
        )
        return holiday_file

    @pytest.fixture
    def mock_config(
        self, tmp_path: Path, sample_holiday_file: Path
    ) -> PreprocessingConfig:
        config = MagicMock(spec=PreprocessingConfig)
        config.temporal_covariates = TemporalCovariatesConfig(
            include_day_of_week=True,
            include_holidays=True,
            holiday_calendar_file=str(sample_holiday_file),
        )
        return config

    def test_processor_output_shape(self, mock_config: PreprocessingConfig):
        processor = TemporalCovariatesProcessor(mock_config)
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2020-01-14")

        result = processor.process(start_date, end_date)

        assert result.shape == (14, 3)
        assert list(result.coords["covariate"].values) == [
            "dow_sin",
            "dow_cos",
            "is_holiday",
        ]

    def test_day_of_week_encoding(self, mock_config: PreprocessingConfig):
        processor = TemporalCovariatesProcessor(mock_config)

        result = processor.process(
            pd.Timestamp("2020-01-06"),
            pd.Timestamp("2020-01-12"),
        )

        dow_sin = result.sel(covariate="dow_sin").values
        dow_cos = result.sel(covariate="dow_cos").values

        expected_sin = [math.sin(2 * math.pi * d / 7) for d in range(7)]
        expected_cos = [math.cos(2 * math.pi * d / 7) for d in range(7)]

        # float16 has ~3-4 decimal digits precision
        np.testing.assert_allclose(dow_sin, expected_sin, rtol=1e-3)
        np.testing.assert_allclose(dow_cos, expected_cos, rtol=1e-3)

    def test_holiday_indicator(self, mock_config: PreprocessingConfig):
        processor = TemporalCovariatesProcessor(mock_config)

        result = processor.process(
            pd.Timestamp("2020-12-24"),
            pd.Timestamp("2020-12-26"),
        )

        is_holiday = result.sel(covariate="is_holiday").values

        assert is_holiday[0] == 0.0
        assert is_holiday[1] == 1.0
        assert is_holiday[2] == 0.0

    def test_no_holidays_in_range(self, mock_config: PreprocessingConfig):
        processor = TemporalCovariatesProcessor(mock_config)

        result = processor.process(
            pd.Timestamp("2020-02-01"),
            pd.Timestamp("2020-02-07"),
        )

        is_holiday = result.sel(covariate="is_holiday").values
        assert np.all(is_holiday == 0.0)

    def test_cyclic_continuity_at_week_boundary(self, mock_config: PreprocessingConfig):
        processor = TemporalCovariatesProcessor(mock_config)

        result = processor.process(
            pd.Timestamp("2020-01-05"),
            pd.Timestamp("2020-01-19"),
        )

        dow_sin = result.sel(covariate="dow_sin").values

        assert len(dow_sin) == 15

        first_sunday_sin = dow_sin[0]
        second_sunday_sin = dow_sin[7]

        np.testing.assert_allclose(first_sunday_sin, second_sunday_sin, atol=1e-6)

    def test_alignment_with_known_dates(self, mock_config: PreprocessingConfig):
        processor = TemporalCovariatesProcessor(mock_config)

        result = processor.process(
            pd.Timestamp("2020-09-11"),
            pd.Timestamp("2020-09-11"),
        )

        is_holiday = result.sel(covariate="is_holiday", date="2020-09-11").item()
        assert is_holiday == 1.0

    def test_catalonia_holiday_file_exists(self):
        holiday_path = Path("data/files/catalonia_holidays.csv")
        assert holiday_path.exists(), "Catalonia holidays file should exist"

        df = pd.read_csv(holiday_path)
        assert "date" in df.columns
        assert len(df) >= 80

        dates = pd.to_datetime(df["date"])
        assert dates.min().year <= 2020
        assert dates.max().year >= 2025


class TestTemporalCovariatesConstants:
    def test_temporal_covariate_dim_constant(self):
        assert TEMPORAL_COVARIATE_DIM == 3

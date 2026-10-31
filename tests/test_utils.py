import pytest
import xarray as xr
import pandas as pd
from unittest.mock import patch

from utils.platform import stage_dataset_to_nvme
from utils.temporal import (
    parse_date_string,
    date_to_index,
    get_temporal_boundaries,
    validate_temporal_range,
)

# --- Test Platform ---


class TestPlatform:
    @patch("utils.platform.shutil.which")
    @patch("subprocess.run")
    def test_stage_dataset_to_nvme_rsync(self, mock_run, mock_which, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        nvme = tmp_path / "nvme"
        nvme.mkdir()

        mock_which.return_value = "/usr/bin/rsync"

        with patch("utils.platform.get_nvme_path", return_value=nvme):
            dest = stage_dataset_to_nvme(source, enable_staging=True)

            assert dest == nvme / "source"
            mock_run.assert_called()
            args = mock_run.call_args[0][0]
            assert "rsync" in args


# --- Test Temporal ---


class TestTemporal:
    @pytest.fixture
    def mock_dataset(self):
        times = pd.date_range("2020-01-01", periods=10)
        ds = xr.Dataset(coords={"date": times})
        # Mock TEMPORAL_COORD constant logic by ensuring code uses "date" or we patch it
        # The utils module imports TEMPORAL_COORD.
        # We assume TEMPORAL_COORD is "date" or similar.
        # Let's patch TEMPORAL_COORD in utils.temporal
        return ds

    def test_parse_date_string(self):
        dt = parse_date_string("2020-01-01")
        assert dt.year == 2020
        assert dt.month == 1
        assert dt.day == 1

        with pytest.raises(ValueError):
            parse_date_string("invalid")

    def test_date_to_index(self, mock_dataset):
        with patch("utils.temporal.TEMPORAL_COORD", "date"):
            idx = date_to_index(mock_dataset, pd.Timestamp("2020-01-02"))
            assert idx == 1

            with pytest.raises(ValueError):
                date_to_index(mock_dataset, pd.Timestamp("2021-01-01"))

    def test_get_temporal_boundaries(self, mock_dataset):
        with patch("utils.temporal.TEMPORAL_COORD", "date"):
            # 0..9 indices
            # Train end: 2020-01-04 (idx 3) -> Train [0, 3)
            # Val end: 2020-01-06 (idx 5) -> Val [3, 5)
            # Test end: 2020-01-08 (idx 7) -> Test [5, 7)

            s, te, ve, tste = get_temporal_boundaries(
                mock_dataset, "2020-01-04", "2020-01-06", "2020-01-08"
            )
            assert s == 0
            assert te == 3
            assert ve == 5
            assert tste == 7

    def test_validate_temporal_range(self):
        # Valid
        validate_temporal_range(
            (0, 10), input_window_length=5, forecast_horizon=2, total_time_steps=20
        )

        # Invalid: range too small
        with pytest.raises(ValueError, match="requires at least"):
            validate_temporal_range(
                (0, 5), input_window_length=5, forecast_horizon=2, total_time_steps=20
            )

        # Invalid: out of bounds
        with pytest.raises(ValueError, match="out of bounds"):
            validate_temporal_range(
                (0, 25), input_window_length=5, forecast_horizon=2, total_time_steps=20
            )

import pytest
import numpy as np
import pandas as pd
import torch


@pytest.fixture
def accelerator_device():
    """Return available accelerator (cuda/mps) or cpu for cross-device testing.

    Tests using this fixture should be marked with @pytest.mark.device to allow
    selective execution of device-dependent tests.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def spawn_builders():
    """Request the real spawned builder pool instead of the in-process one."""


@pytest.fixture(autouse=True)
def in_process_builder_pool(request, monkeypatch):
    """Run heavy-state builds in threads of the test process.

    The production pool spawns interpreters that each import torch (seconds per
    pool); the worker function and its results are identical in a thread, so
    only tests that request ``spawn_builders`` pay for real processes.
    """
    if "spawn_builders" in request.fixturenames:
        return
    from concurrent.futures import ThreadPoolExecutor

    from data import block_pipeline

    monkeypatch.setattr(
        block_pipeline,
        "spawn_builder_pool",
        lambda block_builders, builder_threads: ThreadPoolExecutor(
            max_workers=block_builders
        ),
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "device: tests that require accelerator device (cuda/mps)"
    )


@pytest.fixture(params=["sparse", "dense", "gap_heavy"])
def synthetic_time_series(request):
    """
    Parametrized fixture providing different types of synthetic time series.
    Returns a pandas Series with a date index.
    """
    dates = pd.date_range("2022-01-01", periods=60, freq="D")
    rng = np.random.default_rng(42)

    # Base signal (log-normal)
    values = rng.lognormal(mean=2, sigma=0.5, size=60)

    mode = request.param
    if mode == "sparse":
        # Only 20% of data present
        mask = rng.random(60) < 0.2
        values[~mask] = np.nan
    elif mode == "dense":
        # 95% of data present
        mask = rng.random(60) < 0.95
        values[~mask] = np.nan
    elif mode == "gap_heavy":
        # Large gaps in data
        values[10:25] = np.nan
        values[40:55] = np.nan

    return pd.Series(values, index=dates)

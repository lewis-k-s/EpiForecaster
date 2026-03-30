from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch
from torch_geometric.data import Batch

import plotting.forecast_plots as forecast_plots
from data.epi_batch import EpiBatch
from evaluation.selection import (
    WindowSelectionSpec,
    load_window_selection_specs_from_granular,
    select_windows_by_loss,
)


def test_collect_forecast_samples_supports_epibatch_attribute_targets(
    monkeypatch,
) -> None:
    class _DummyDataset:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                model=SimpleNamespace(input_window_length=3, forecast_horizon=2)
            )
            self.precomputed_cases_hist = torch.zeros((10, 1), dtype=torch.float32)
            self.precomputed_hosp = torch.arange(10, dtype=torch.float32).reshape(10, 1)
            self._index_lookup = {(0, 5): 0}
            self.dataset = {"date": SimpleNamespace(values=list(range(10)))}

        def num_windows(self) -> int:
            return 1

        def get_valid_window_starts_dict(self, *, mode: str, required_targets: list[str]):
            assert mode == "all"
            assert required_targets == ["hospitalizations"]
            return {0: [5]}

        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int):
            raise AssertionError("Fake DataLoader should bypass dataset indexing")

    dummy_dataset = _DummyDataset()
    monkeypatch.setattr(forecast_plots, "EpiDataset", _DummyDataset)

    batch = EpiBatch(
        hosp_hist=torch.zeros((1, 3, 3), dtype=torch.float32),
        deaths_hist=torch.zeros((1, 3, 3), dtype=torch.float32),
        cases_hist=torch.zeros((1, 3, 3), dtype=torch.float32),
        bio_node=torch.zeros((1, 3, 1), dtype=torch.float32),
        mob_batch=Batch(),
        population=torch.ones((1,), dtype=torch.float32),
        b=1,
        t=3,
        target_node=torch.tensor([0], dtype=torch.long),
        target_region_index=None,
        window_start=torch.tensor([5], dtype=torch.long),
        node_labels=["Region 0"],
        temporal_covariates=torch.zeros((1, 3, 0), dtype=torch.float32),
        ww_hist=torch.zeros((1, 3), dtype=torch.float32),
        ww_hist_mask=torch.zeros((1, 3), dtype=torch.float32),
        hosp_target=torch.tensor([[8.0, 9.0]], dtype=torch.float32),
        ww_target=torch.zeros((1, 2), dtype=torch.float32),
        cases_target=torch.zeros((1, 2), dtype=torch.float32),
        deaths_target=torch.zeros((1, 2), dtype=torch.float32),
        hosp_target_mask=torch.ones((1, 2), dtype=torch.float32),
        ww_target_mask=torch.zeros((1, 2), dtype=torch.float32),
        cases_target_mask=torch.zeros((1, 2), dtype=torch.float32),
        deaths_target_mask=torch.zeros((1, 2), dtype=torch.float32),
    )

    class _FakeDataLoader:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __iter__(self):
            yield batch

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))

        def forward_batch(self, *, batch_data, region_embeddings=None):
            return {
                "pred_hosp": torch.tensor([[7.5, 9.5]], dtype=torch.float32),
            }, {}

    monkeypatch.setattr(forecast_plots, "DataLoader", _FakeDataLoader)
    monkeypatch.setattr(forecast_plots, "prepare_batch_for_device", lambda batch, **_: batch)

    loader = SimpleNamespace(dataset=dummy_dataset)
    model = _DummyModel()

    samples = forecast_plots.collect_forecast_samples_for_target_nodes(
        target_node_ids=[0],
        model=model,
        loader=loader,
        target_names=["hosp"],
        required_targets=["hosp"],
    )

    assert len(samples) == 1
    hosp_payload = samples[0]["targets"]["hosp"]
    assert hosp_payload["target"].tolist() == [8.0, 9.0]
    assert hosp_payload["prediction"].tolist() == [7.5, 9.5]
    assert hosp_payload["window_mae"] == 0.5
    assert samples[0]["window_mae"] == 0.5


def test_load_window_selection_specs_from_granular_aggregates_per_target_means(
    tmp_path,
) -> None:
    granular_csv = tmp_path / "granular.csv"
    pd.DataFrame(
        [
            {"split": "test", "target": "cases", "node_id": 1, "window_start": 10, "abs_error": 1.0},
            {"split": "test", "target": "cases", "node_id": 1, "window_start": 10, "abs_error": 3.0},
            {"split": "test", "target": "hospitalizations", "node_id": 1, "window_start": 10, "abs_error": 10.0},
            {"split": "test", "target": "cases", "node_id": 2, "window_start": 11, "abs_error": 5.0},
        ]
    ).to_csv(granular_csv, index=False)

    specs = load_window_selection_specs_from_granular(
        granular_csv=granular_csv,
        split="test",
    )

    assert len(specs) == 2
    spec = next(spec for spec in specs if spec.node_id == 1 and spec.window_start == 10)
    assert spec.score == 6.0
    assert spec.observed_targets == ("cases", "hospitalizations")
    assert spec.observed_points == 3


def test_select_windows_by_loss_returns_window_specs() -> None:
    specs = [
        WindowSelectionSpec(i, 100 + i, float(i), ("cases",), 2)
        for i in range(8)
    ]
    groups = select_windows_by_loss(window_specs=specs, samples_per_group=1)
    assert set(groups) == {
        "Q1 (Best MAE)",
        "Q2 (Good MAE)",
        "Q3 (Poor MAE)",
        "Q4 (Worst MAE)",
    }
    assert all(len(group) == 1 for group in groups.values())
    assert all(isinstance(group[0], WindowSelectionSpec) for group in groups.values())


def test_collect_forecast_samples_for_window_specs_uses_exact_window(
    monkeypatch,
) -> None:
    class _DummyDataset:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                model=SimpleNamespace(input_window_length=3, forecast_horizon=2)
            )
            self.precomputed_cases_hist = torch.zeros((10, 1), dtype=torch.float32)
            self.precomputed_hosp = torch.arange(10, dtype=torch.float32).reshape(10, 1)
            self._index_lookup = {(0, 4): 0}
            self._temporal_coords = list(range(10))

        def index_for_target_node_window(
            self, *, target_node: int, window_idx: int
        ) -> int:
            raise AssertionError(
                "exact-window collection should use absolute window starts, not ordinal indices"
            )

    dummy_dataset = _DummyDataset()
    monkeypatch.setattr(forecast_plots, "EpiDataset", _DummyDataset)

    batch = EpiBatch(
        hosp_hist=torch.zeros((1, 3, 3), dtype=torch.float32),
        deaths_hist=torch.zeros((1, 3, 3), dtype=torch.float32),
        cases_hist=torch.zeros((1, 3, 3), dtype=torch.float32),
        bio_node=torch.zeros((1, 3, 1), dtype=torch.float32),
        mob_batch=Batch(),
        population=torch.ones((1,), dtype=torch.float32),
        b=1,
        t=3,
        target_node=torch.tensor([0], dtype=torch.long),
        target_region_index=None,
        window_start=torch.tensor([4], dtype=torch.long),
        node_labels=["Region 0"],
        temporal_covariates=torch.zeros((1, 3, 0), dtype=torch.float32),
        ww_hist=torch.zeros((1, 3), dtype=torch.float32),
        ww_hist_mask=torch.zeros((1, 3), dtype=torch.float32),
        hosp_target=torch.tensor([[7.0, 8.0]], dtype=torch.float32),
        ww_target=torch.zeros((1, 2), dtype=torch.float32),
        cases_target=torch.zeros((1, 2), dtype=torch.float32),
        deaths_target=torch.zeros((1, 2), dtype=torch.float32),
        hosp_target_mask=torch.ones((1, 2), dtype=torch.float32),
        ww_target_mask=torch.zeros((1, 2), dtype=torch.float32),
        cases_target_mask=torch.zeros((1, 2), dtype=torch.float32),
        deaths_target_mask=torch.zeros((1, 2), dtype=torch.float32),
    )

    class _FakeDataLoader:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __iter__(self):
            yield batch

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))

        def forward_batch(self, *, batch_data, region_embeddings=None):
            return {
                "pred_hosp": torch.tensor([[6.5, 8.5]], dtype=torch.float32),
            }, {}

    monkeypatch.setattr(forecast_plots, "DataLoader", _FakeDataLoader)
    monkeypatch.setattr(forecast_plots, "prepare_batch_for_device", lambda batch, **_: batch)

    loader = SimpleNamespace(dataset=dummy_dataset)
    model = _DummyModel()
    samples = forecast_plots.collect_forecast_samples_for_window_specs(
        window_specs=[
            WindowSelectionSpec(
                node_id=0,
                window_start=4,
                score=0.5,
                observed_targets=("hospitalizations",),
                observed_points=2,
            )
        ],
        model=model,
        loader=loader,
        target_names=["hosp"],
    )

    assert len(samples) == 1
    assert samples[0]["window_start"] == 4
    assert samples[0]["start_time"] == "4"
    assert samples[0]["targets"]["hosp"]["target"].tolist() == [7.0, 8.0]
    assert samples[0]["targets"]["hosp"]["window_mae"] == 0.5


def test_make_forecast_figure_annotates_window_mae() -> None:
    fig = forecast_plots.make_forecast_figure(
        samples=[
            {
                "node_id": 1,
                "node_label": "Region 1",
                "actual_context": [1.0, 2.0, 3.0, 4.0],
                "prediction": [3.0, 5.0],
                "target": [3.0, 4.0],
                "history": [1.0, 2.0],
                "t_rel": [-2, -1, 0, 1],
                "t0_idx_in_context": 2,
                "start_time": "2024-01-01",
                "window_start": 10,
                "window_mae": 0.5,
                "L": 2,
                "H": 2,
            }
        ],
        input_window_length=2,
        forecast_horizon=2,
        target=None,
        target_label="Hospitalizations",
    )
    assert fig is not None
    title = fig.axes[0].get_title()
    assert "MAE=0.500" in title


def test_make_forecast_figure_supports_shared_labels() -> None:
    fig = forecast_plots.make_forecast_figure(
        samples=[
            {
                "node_id": 1,
                "node_label": "Region 1",
                "actual_context": [1.0, 2.0, 3.0, 4.0],
                "prediction": [3.0, 5.0],
                "target": [3.0, 4.0],
                "history": [1.0, 2.0],
                "t_rel": [-2, -1, 0, 1],
                "t0_idx_in_context": 2,
                "start_time": "2024-01-01",
                "window_start": 10,
                "window_mae": 0.5,
                "L": 2,
                "H": 2,
            }
        ],
        input_window_length=2,
        forecast_horizon=2,
        target=None,
        target_label="Hospitalizations",
        figure_title="Cases (MAE)",
        shared_xlabel="Time (days relative to forecast start)",
    )
    assert fig is not None
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Cases (MAE)"
    assert fig.axes[0].get_xlabel() == ""
    assert any(
        text.get_text() == "Time (days relative to forecast start)"
        for text in fig.texts
    )

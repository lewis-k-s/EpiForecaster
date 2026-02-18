import pytest
import torch
import numpy as np
import xarray as xr
from unittest.mock import patch
from models.configs import EpiForecasterConfig, DataConfig, ModelConfig, ModelVariant
from data.epi_dataset import EpiDataset, collate_epiforecaster_batch


class TestEpiDataset:
    @pytest.fixture
    def mock_xarray_dataset(self):
        T, N = 30, 5
        times = np.arange(T)
        regions = np.arange(N).astype(str)

        # Clinical vars
        hosp = np.random.rand(T, N)
        hosp_mask = np.ones((T, N))
        hosp_age = np.zeros((T, N))

        # Mobility (T, N, N)
        mobility = np.random.rand(T, N, N)
        temporal_covariates = np.random.rand(T, 3)
        temporal_covariates[3, 1] = np.nan

        # Population
        pop = np.ones(N) * 1000

        # Use region_id for coordinate as expected by EpiDataset
        ds = xr.Dataset(
            {
                "hospitalizations": (("date", "region_id"), hosp),
                "hospitalizations_mask": (("date", "region_id"), hosp_mask),
                "hospitalizations_age": (("date", "region_id"), hosp_age),
                "deaths": (("date", "region_id"), hosp),
                "deaths_mask": (("date", "region_id"), hosp_mask),
                "deaths_age": (("date", "region_id"), hosp_age),
                "cases": (("date", "region_id"), hosp),
                "cases_mask": (("date", "region_id"), hosp_mask),
                "cases_age": (("date", "region_id"), hosp_age),
                "mobility": (("date", "origin", "target"), mobility),
                "temporal_covariates": (("date", "covariate"), temporal_covariates),
                "population": (("region_id"), pop),
                "run_id": "real",
                "synthetic_sparsity_level": 0.0,
            },
            coords={
                "date": times,
                "region_id": regions,
                "origin": regions,
                "target": regions,
                "covariate": np.arange(3),
            },
        )
        return ds

    @pytest.fixture
    def config(self):
        return EpiForecasterConfig(
            data=DataConfig(
                dataset_path="dummy.zarr", run_id="real", mobility_threshold=0.0
            ),
            model=ModelConfig(
                type=ModelVariant(cases=True, mobility=True),
                history_length=5,
                forecast_horizon=3,
                mobility_embedding_dim=8,
                region_embedding_dim=8,
                max_neighbors=5,
                gnn_depth=1,
                gnn_module="gcn",
            ),
        )

    def test_initialization(self, config, mock_xarray_dataset):
        with patch.object(
            EpiDataset, "load_canonical_dataset", return_value=mock_xarray_dataset
        ):
            ds = EpiDataset(
                config=config,
                target_nodes=[0, 1],
                context_nodes=[0, 1, 2],
            )
            assert ds.num_nodes == 5
            assert len(ds) > 0

    def test_getitem_shapes(self, config, mock_xarray_dataset):
        with patch.object(
            EpiDataset, "load_canonical_dataset", return_value=mock_xarray_dataset
        ):
            ds = EpiDataset(
                config=config,
                target_nodes=[0],
                context_nodes=[0, 1],
            )

            item = ds[0]

            L = config.model.history_length

            assert item["hosp_hist"].shape == (L, 3)
            assert item["mob_x"].shape[2] >= 9

            assert len(item["mob_edge_index"]) == L

    def test_feature_masking(self, config, mock_xarray_dataset):
        with patch.object(
            EpiDataset, "load_canonical_dataset", return_value=mock_xarray_dataset
        ):
            ds = EpiDataset(
                config=config,
                target_nodes=[0],
                context_nodes=[0, 1, 2, 3, 4],
            )

            assert len(ds._precomputed_k_hop_masks) > 0

            item = ds[0]
            assert item["mob_x"].shape[1] == ds.num_nodes

    def test_invalid_window(self, config, mock_xarray_dataset):
        with patch.object(
            EpiDataset, "load_canonical_dataset", return_value=mock_xarray_dataset
        ):
            ds = EpiDataset(
                config=config,
                target_nodes=[0],
                context_nodes=[0],
            )

            with pytest.raises(IndexError):
                ds[len(ds) + 100]

    def test_no_mobility_config(self, config, mock_xarray_dataset):
        config.model.type.mobility = False
        with patch.object(
            EpiDataset, "load_canonical_dataset", return_value=mock_xarray_dataset
        ):
            ds = EpiDataset(
                config=config,
                target_nodes=[0],
                context_nodes=[0],
            )
            assert ds.preloaded_mobility is not None

    def test_value_channels_are_finite_after_getitem(self, config, mock_xarray_dataset):
        with patch.object(
            EpiDataset, "load_canonical_dataset", return_value=mock_xarray_dataset
        ):
            ds = EpiDataset(
                config=config,
                target_nodes=[0],
                context_nodes=[0, 1, 2],
            )
            item = ds[0]

            assert torch.isfinite(item["hosp_hist"][..., 0]).all()
            assert torch.isfinite(item["deaths_hist"][..., 0]).all()
            assert torch.isfinite(item["cases_hist"][..., 0]).all()
            assert torch.isfinite(item["bio_node"][..., 0::4]).all()
            assert torch.isfinite(item["temporal_covariates"]).all()
            assert torch.isfinite(item["mob_x"]).all()

    def test_collate_sanitizes_non_finite_value_channels(
        self, config, mock_xarray_dataset
    ):
        with patch.object(
            EpiDataset, "load_canonical_dataset", return_value=mock_xarray_dataset
        ):
            ds = EpiDataset(
                config=config,
                target_nodes=[0, 1],
                context_nodes=[0, 1, 2],
            )
            item_a = ds[0]
            item_b = ds[1]

            item_a["hosp_hist"][0, 0] = float("nan")
            item_a["deaths_hist"][0, 0] = float("inf")
            item_a["cases_hist"][0, 0] = float("-inf")
            item_a["bio_node"][0, 0] = float("nan")
            item_a["temporal_covariates"][0, 0] = float("nan")
            item_a["mob_x"][0, 0, 0] = float("inf")
            item_a["mob_edge_weight"][0][0] = float("nan")

            batch = collate_epiforecaster_batch([item_a, item_b], require_region_index=False)

            assert torch.isfinite(batch["HospHist"]).all()
            assert torch.isfinite(batch["DeathsHist"]).all()
            assert torch.isfinite(batch["CasesHist"]).all()
            assert torch.isfinite(batch["BioNode"]).all()
            assert torch.isfinite(batch["TemporalCovariates"]).all()
            assert torch.isfinite(batch["MobBatch"].x).all()
            assert torch.isfinite(batch["MobBatch"].edge_weight).all()

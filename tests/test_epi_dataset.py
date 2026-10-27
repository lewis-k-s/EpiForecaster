import pytest
import torch
import numpy as np
import xarray as xr
from unittest.mock import patch
from models.configs import EpiForecasterConfig, DataConfig, ModelConfig, ModelVariant
from data.epi_dataset import SHARED_SERIES_ATTRS, EpiDataset
from data.epi_batch import collate_epiforecaster_batch
from utils.precision_policy import MODEL_PARAM_DTYPE
from tests.dataset_test_helpers import context_features_for_item


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
                dataset_path="dummy.zarr",
                run_id="real",
                mobility_min_flow_raw=0.0,
            ),
            model=ModelConfig(
                type=ModelVariant(cases=True, mobility=True),
                input_window_length=5,
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

    def test_temporal_scaler_excludes_future_values_and_is_reused(
        self, config, mock_xarray_dataset
    ):
        dataset = mock_xarray_dataset.copy(deep=True)
        values = np.full((30, 5), 2.0, dtype=np.float32)
        values[15:] = 1000.0
        name = "edar_biomarker_N1"
        for suffix, data in (
            ("", values),
            ("_mask", np.ones_like(values)),
            ("_censor", np.zeros_like(values)),
            ("_age", np.zeros_like(values)),
        ):
            dataset[f"{name}{suffix}"] = (("date", "region_id"), data)
        with patch.object(EpiDataset, "load_canonical_dataset", return_value=dataset):
            train = EpiDataset(
                config=config,
                target_nodes=[0, 1],
                context_nodes=[0, 1],
                time_range=(0, 15),
            )
            validation = EpiDataset(
                config=config,
                target_nodes=[2],
                context_nodes=[0, 1, 2],
                time_range=(15, 30),
                biomarker_preprocessor=train.biomarker_preprocessor,
                mobility_preprocessor=train.mobility_preprocessor,
            )
        assert train.biomarker_preprocessor.scaler_params.center[name] == 2.0
        assert validation.biomarker_preprocessor is train.biomarker_preprocessor
        assert validation.biomarker_preprocessor.scaler_params.center[name] == 2.0

    def test_supervision_targets_preserve_small_positive_float32_values(
        self, config, mock_xarray_dataset
    ):
        dataset = mock_xarray_dataset.copy(deep=True)
        tiny = np.full(
            dataset["cases"].shape,
            1e-12,
            dtype=np.float32,
        )
        for name in ("cases", "hospitalizations", "deaths"):
            dataset[name] = (("date", "region_id"), tiny.copy())
        for name in (
            "edar_biomarker_N1",
            "edar_biomarker_N2",
            "edar_biomarker_IP4",
        ):
            dataset[name] = (("date", "region_id"), tiny.copy())
            dataset[f"{name}_mask"] = (
                ("date", "region_id"),
                np.ones_like(tiny, dtype=bool),
            )
            dataset[f"{name}_censor"] = (
                ("date", "region_id"),
                np.zeros_like(tiny, dtype=np.uint8),
            )
            dataset[f"{name}_age"] = (
                ("date", "region_id"),
                np.ones_like(tiny, dtype=np.uint8),
            )
        for name in (
            "latent_S_true",
            "latent_I_true",
            "latent_H_true",
            "latent_R_true",
            "latent_D_true",
        ):
            dataset[name] = (("date", "region_id"), tiny.copy())

        with patch.object(EpiDataset, "load_canonical_dataset", return_value=dataset):
            epi_dataset = EpiDataset(
                config=config,
                target_nodes=[0, 1],
                context_nodes=[0, 1, 2],
            )

        assert epi_dataset.precomputed_cases_target.dtype == torch.float32
        assert epi_dataset.precomputed_hosp.dtype == torch.float32
        assert epi_dataset.precomputed_deaths.dtype == torch.float32
        assert epi_dataset.precomputed_ww.dtype == torch.float32
        assert epi_dataset.precomputed_I_target is not None
        assert epi_dataset.precomputed_I_target.dtype == torch.float32
        assert bool((epi_dataset.precomputed_cases_target > 0).all())
        assert bool((epi_dataset.precomputed_ww > 0).all())
        assert bool((epi_dataset.precomputed_I_target > 0).all())

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

            L = config.model.input_window_length

            assert item["hosp_hist"].shape == (L, 3)
            features = context_features_for_item(ds, item)
            assert features.shape[0] == L
            assert features.shape[2] >= 9
            assert int(item["window_start"]) >= 0
            assert ds.context_feature_ww is None

    def test_large_population_remains_finite_and_exact(
        self, config, mock_xarray_dataset
    ):
        populations = np.array([1_686_208, 100_000, 65_505, 65_504, 26])
        mock_xarray_dataset["population"] = (("region_id",), populations)

        with patch.object(
            EpiDataset, "load_canonical_dataset", return_value=mock_xarray_dataset
        ):
            ds = EpiDataset(
                config=config,
                target_nodes=[0],
                context_nodes=[0, 1],
            )

            population = ds.node_static_covariates["Pop"]
            assert population.dtype == torch.float32
            assert torch.isfinite(population).all()
            torch.testing.assert_close(
                population, torch.from_numpy(populations).float()
            )

    def test_feature_masking(self, config, mock_xarray_dataset):
        with patch.object(
            EpiDataset, "load_canonical_dataset", return_value=mock_xarray_dataset
        ):
            ds = EpiDataset(
                config=config,
                target_nodes=[0],
                context_nodes=[0, 1, 2, 3, 4],
            )

            assert ds._target_k_hop_reach is not None

            item = ds[0]
            features = context_features_for_item(ds, item)
            assert features.shape[1] == ds._graph_node_ids.numel()

    def test_shared_series_match_a_fresh_build(self, config, mock_xarray_dataset):
        with patch.object(
            EpiDataset, "load_canonical_dataset", return_value=mock_xarray_dataset
        ):
            train = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])
            shared_kwargs = {
                "config": config,
                "target_nodes": [2],
                "context_nodes": [0, 1, 2],
                "biomarker_preprocessor": train.biomarker_preprocessor,
                "mobility_preprocessor": train.mobility_preprocessor,
                "preloaded_mobility": train.preloaded_mobility,
                "mobility_mask": train.mobility_mask,
            }
            fresh = EpiDataset(**shared_kwargs)
            shared = EpiDataset(**shared_kwargs, shared_series=train.shared_series)

        for name in SHARED_SERIES_ATTRS:
            value = getattr(shared, name)
            # Reused without a copy, and equal to what the dataset would build.
            assert value is getattr(train, name)
            if value is not None:
                assert torch.equal(value, getattr(fresh, name))
        assert shared.temporal_covariates_dim == fresh.temporal_covariates_dim
        assert len(shared) == len(fresh)
        for index in range(len(fresh)):
            for key, value in fresh[index].items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(shared[index][key], value), key

    @pytest.mark.parametrize(
        ("corrupt", "message"),
        [
            ("outside_mask", "zero outside mobility_mask"),
            ("non_finite", "finite and nonnegative"),
            ("negative", "finite and nonnegative"),
        ],
    )
    def test_shared_mobility_rejects_invalid_weights(
        self, config, mock_xarray_dataset, corrupt, message
    ):
        with patch.object(
            EpiDataset, "load_canonical_dataset", return_value=mock_xarray_dataset
        ):
            train = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])
            weights = train.preloaded_mobility.clone()
            mask = train.mobility_mask
            # Corrupt the last edge so a check that stops early would miss it.
            edges = torch.nonzero(~mask if corrupt == "outside_mask" else mask)
            assert edges.numel() > 0
            weights[tuple(edges[-1].tolist())] = {
                "outside_mask": 1.0,
                "non_finite": float("inf"),
                "negative": -1.0,
            }[corrupt]
            with pytest.raises(ValueError, match=message):
                EpiDataset(
                    config=config,
                    target_nodes=[2],
                    context_nodes=[0, 1, 2],
                    biomarker_preprocessor=train.biomarker_preprocessor,
                    preloaded_mobility=weights,
                    mobility_mask=mask,
                )

    def test_shared_series_require_fitted_biomarker_preprocessor(
        self, config, mock_xarray_dataset
    ):
        with patch.object(
            EpiDataset, "load_canonical_dataset", return_value=mock_xarray_dataset
        ):
            train = EpiDataset(config=config, target_nodes=[0], context_nodes=[0])
            with pytest.raises(ValueError, match="biomarker_preprocessor"):
                EpiDataset(
                    config=config,
                    target_nodes=[1],
                    context_nodes=[0, 1],
                    shared_series=train.shared_series,
                )

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
            assert torch.isfinite(context_features_for_item(ds, item)).all()

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

            # Dense features are assembled lazily; dirty table values must be
            ds.context_feature_clinical[0, 0, 0] = float("inf")

            batch = collate_epiforecaster_batch(
                [item_a, item_b], require_region_index=False
            )

            assert torch.isfinite(batch.hosp_hist).all()
            assert torch.isfinite(batch.deaths_hist).all()
            assert torch.isfinite(batch.cases_hist).all()
            assert torch.isfinite(batch.bio_node).all()
            assert torch.isfinite(batch.temporal_covariates).all()

            from utils.training_utils import inject_context_features

            inject_context_features(batch, ds, torch.device("cpu"))
            assert torch.isfinite(batch.mob_batch.x_dense).all()
            assert batch.hosp_hist.dtype == MODEL_PARAM_DTYPE
            assert batch.deaths_hist.dtype == MODEL_PARAM_DTYPE
            assert batch.cases_hist.dtype == MODEL_PARAM_DTYPE
            assert batch.bio_node.dtype == MODEL_PARAM_DTYPE
            assert batch.population.dtype == MODEL_PARAM_DTYPE
            assert batch.temporal_covariates.dtype == MODEL_PARAM_DTYPE
            assert batch.ww_hist.dtype == MODEL_PARAM_DTYPE
            assert batch.ww_hist_mask.dtype == MODEL_PARAM_DTYPE
            assert batch.hosp_target.dtype == MODEL_PARAM_DTYPE
            assert batch.ww_target.dtype == MODEL_PARAM_DTYPE
            assert batch.cases_target.dtype == MODEL_PARAM_DTYPE
            assert batch.deaths_target.dtype == MODEL_PARAM_DTYPE
            assert batch.hosp_target_mask.dtype == MODEL_PARAM_DTYPE
            assert batch.ww_target_mask.dtype == MODEL_PARAM_DTYPE
            assert batch.cases_target_mask.dtype == MODEL_PARAM_DTYPE
            assert batch.deaths_target_mask.dtype == MODEL_PARAM_DTYPE
            assert batch.mob_batch.x_dense.dtype == MODEL_PARAM_DTYPE

    def test_collate_reuses_shared_mob_real_node_idx(self, config, mock_xarray_dataset):
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

            batch = collate_epiforecaster_batch(
                [item_a, item_b], require_region_index=False
            )

            assert batch.mob_batch.mob_real_node_idx is item_a["mob_real_node_idx"]

    def test_collate_rejects_inconsistent_mob_real_node_idx(
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
            item_b["mob_real_node_idx"] = item_b["mob_real_node_idx"].clone()

            with pytest.raises(ValueError, match="same cached tensor"):
                collate_epiforecaster_batch(
                    [item_a, item_b], require_region_index=False
                )

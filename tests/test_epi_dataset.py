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

            batch = collate_epiforecaster_batch(
                [item_a, item_b], require_region_index=False
            )

            assert torch.isfinite(batch["HospHist"]).all()
            assert torch.isfinite(batch["DeathsHist"]).all()
            assert torch.isfinite(batch["CasesHist"]).all()
            assert torch.isfinite(batch["BioNode"]).all()
            assert torch.isfinite(batch["TemporalCovariates"]).all()
            assert torch.isfinite(batch["MobBatch"].x).all()
            assert torch.isfinite(batch["MobBatch"].edge_weight).all()

    def test_sparse_graphs_precomputed_at_init(self, config, mock_xarray_dataset):
        """Verify that sparse graphs are eagerly precomputed during initialization."""
        with patch.object(
            EpiDataset, "load_canonical_dataset", return_value=mock_xarray_dataset
        ):
            ds = EpiDataset(
                config=config,
                target_nodes=[0, 1],
                context_nodes=[0, 1, 2],
            )

            # Check that all timesteps have precomputed graphs
            T = len(mock_xarray_dataset.date)
            assert len(ds._full_graph_cache) == T, (
                f"Expected {T} precomputed graphs, got {len(ds._full_graph_cache)}"
            )

            # Verify each graph has valid structure
            for time_step, graph in ds._full_graph_cache.items():
                assert graph.edge_index is not None
                assert graph.edge_weight is not None
                assert graph.num_nodes > 0
                assert graph.node_ids is not None
                # edge_index should be (2, E)
                assert graph.edge_index.shape[0] == 2
                # edge_weight should match number of edges
                assert graph.edge_weight.shape[0] == graph.edge_index.shape[1]

    def test_full_sparse_prune_matches_direct_dense_mask(
        self, config, mock_xarray_dataset
    ):
        """Pruned shared sparse topology should match direct dense masking output."""
        with patch.object(
            EpiDataset, "load_canonical_dataset", return_value=mock_xarray_dataset
        ):
            ds = EpiDataset(
                config=config,
                target_nodes=[0],
                context_nodes=[0, 1, 2],
            )

            node_mask = ds._get_graph_node_mask()
            node_ids = torch.where(node_mask)[0]
            global_to_local = torch.full((ds.num_nodes,), -1, dtype=torch.long)
            global_to_local[node_ids] = torch.arange(node_ids.numel(), dtype=torch.long)

            for time_step, graph in ds._full_graph_cache.items():
                dense = ds.preloaded_mobility[time_step].clone()
                dense[~node_mask] = 0
                dense[:, ~node_mask] = 0

                edge_mask = dense > 0
                origins, destinations = torch.where(edge_mask)
                expected_edge_weight = dense[origins, destinations]
                expected_edge_index = torch.stack(
                    [global_to_local[origins], global_to_local[destinations]], dim=0
                )

                assert torch.equal(graph.node_ids, node_ids)
                assert torch.equal(graph.edge_index, expected_edge_index)
                assert torch.allclose(graph.edge_weight, expected_edge_weight)

    def test_shared_sparse_topology_built_once_when_reused(
        self, config, mock_xarray_dataset
    ):
        """Dense->sparse full topology conversion should happen once with reuse."""

        def _dataset_factory(*_args, **_kwargs):
            return mock_xarray_dataset.copy(deep=True)

        builder = EpiDataset._build_full_sparse_topology
        with patch.object(EpiDataset, "load_canonical_dataset", side_effect=_dataset_factory):
            with patch.object(
                EpiDataset,
                "_build_full_sparse_topology",
                autospec=True,
                side_effect=builder,
            ) as build_spy:
                train_ds = EpiDataset(
                    config=config,
                    target_nodes=[0, 1],
                    context_nodes=[0, 1],
                )
                val_ds = EpiDataset(
                    config=config,
                    target_nodes=[2],
                    context_nodes=[0, 1, 2],
                    biomarker_preprocessor=train_ds.biomarker_preprocessor,
                    mobility_preprocessor=train_ds.mobility_preprocessor,
                    preloaded_mobility=train_ds.preloaded_mobility,
                    mobility_mask=train_ds.mobility_mask,
                    shared_sparse_topology=train_ds.shared_sparse_topology,
                )
                _ = EpiDataset(
                    config=config,
                    target_nodes=[3],
                    context_nodes=[0, 1, 2, 3],
                    biomarker_preprocessor=train_ds.biomarker_preprocessor,
                    mobility_preprocessor=train_ds.mobility_preprocessor,
                    preloaded_mobility=train_ds.preloaded_mobility,
                    mobility_mask=train_ds.mobility_mask,
                    shared_sparse_topology=train_ds.shared_sparse_topology,
                )

                # Full dense->sparse build happens once on the train dataset only.
                assert build_spy.call_count == 1
                assert train_ds.shared_sparse_topology is not None
                assert val_ds.shared_sparse_topology is train_ds.shared_sparse_topology

    def test_create_temporal_splits_reuses_full_sparse_topology(
        self, config, mock_xarray_dataset
    ):
        """Temporal split creation should build full sparse once and prune per split."""

        def _dataset_factory(*_args, **_kwargs):
            return mock_xarray_dataset.copy(deep=True)

        builder = EpiDataset._build_full_sparse_topology
        with patch.object(EpiDataset, "load_canonical_dataset", side_effect=_dataset_factory):
            with patch("utils.temporal.get_temporal_boundaries", return_value=(0, 20, 25, 30)):
                with patch("utils.temporal.validate_temporal_range", return_value=None):
                    with patch(
                        "utils.temporal.format_date_range",
                        return_value="dummy-range",
                    ):
                        with patch.object(
                            EpiDataset,
                            "_build_full_sparse_topology",
                            autospec=True,
                            side_effect=builder,
                        ) as build_spy:
                            train_ds, val_ds, test_ds = EpiDataset.create_temporal_splits(
                                config=config,
                                train_end_date="2020-01-15",
                                val_end_date="2020-01-25",
                                test_end_date="2020-01-30",
                            )

        assert build_spy.call_count == 1
        assert train_ds.shared_sparse_topology is None
        assert val_ds.shared_sparse_topology is None
        assert test_ds.shared_sparse_topology is None
        assert len(train_ds._full_graph_cache) > 0
        assert len(val_ds._full_graph_cache) > 0
        assert len(test_ds._full_graph_cache) > 0

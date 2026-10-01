import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from training.epiforecaster_trainer import (
    EpiForecasterTrainer,
    compute_mean_target_crps,
    compute_mean_target_mae,
    compute_mean_target_nll,
    compute_validation_selection_score,
    normalize_training_history,
)
from models.configs import (
    EpiForecasterConfig,
    ModelConfig,
    DataConfig,
    TrainingParams,
    OutputConfig,
    ModelVariant,
    SIRPhysicsConfig,
    ObservationHeadConfig,
    LossConfig,
    CurriculumConfig,
)
from data.latent_supervision import LATENT_SCALE_TASKS, LatentScaleState


def _latent_scale_state() -> LatentScaleState:
    return LatentScaleState(
        schema_version=1,
        scales={name: np.float32(0.5) for name in LATENT_SCALE_TASKS},
        std={name: 0.5 for name in LATENT_SCALE_TASKS},
        count={name: 10 for name in LATENT_SCALE_TASKS},
        zero_count={name: 0 for name in LATENT_SCALE_TASKS},
        floors={name: 1.0e-8 for name in LATENT_SCALE_TASKS},
        run_ids=("synthetic-a",),
        target_node_hash="nodes",
        supervised_date_hash="dates",
        membership_hash="membership",
    )


def _mock_canonical_region_coordinate(
    mock_dataset_class: MagicMock,
    *,
    total_regions: int = 10,
) -> None:
    aligned_dataset = MagicMock()
    region_coordinate = MagicMock()
    region_coordinate.size = total_regions
    region_coordinate.values = np.asarray(
        [f"{index:03d}" for index in range(total_regions)],
        dtype=object,
    )
    aligned_dataset.__getitem__.side_effect = (
        lambda key: region_coordinate if key == "region_id" else MagicMock()
    )
    aligned_dataset.__contains__.return_value = False
    aligned_dataset.data_vars = {}
    mock_dataset_class.load_canonical_dataset.return_value = aligned_dataset


def test_persist_mobility_diagnostics_writes_all_splits(tmp_path) -> None:
    trainer = EpiForecasterTrainer.__new__(EpiForecasterTrainer)
    trainer.train_dataset = SimpleNamespace(
        mobility_diagnostics={"run_id": "train", "positive_edge_count": 3}
    )
    trainer.val_dataset = SimpleNamespace(
        mobility_diagnostics={"run_id": "validation", "positive_edge_count": 2}
    )
    trainer.test_dataset = SimpleNamespace(
        mobility_diagnostics={"run_id": "test", "positive_edge_count": 1}
    )

    trainer._persist_mobility_diagnostics(tmp_path)

    payload = json.loads(
        (tmp_path / "mobility_initialization_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["train"][0]["positive_edge_count"] == 3
    assert payload["validation"][0]["positive_edge_count"] == 2
    assert payload["test"][0]["positive_edge_count"] == 1


def test_persist_latent_scale_state_refuses_identity_change(tmp_path) -> None:
    trainer = EpiForecasterTrainer.__new__(EpiForecasterTrainer)
    trainer.latent_scale_state = _latent_scale_state()
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(split_seed=42),
        to_dict=lambda: {
            "data": {"dataset_path": str(tmp_path / "dataset.zarr")},
            "training": {"split_seed": 42},
        },
    )

    trainer._persist_latent_scale_state(tmp_path)

    payload = json.loads(
        (tmp_path / "latent_scale_receipt.json").read_text(encoding="utf-8")
    )
    assert payload["state_hash"] == trainer.latent_scale_state.state_hash
    assert len(payload["receipt_hash"]) == 64
    payload["scales"]["i"] = 1.0
    (tmp_path / "latent_scale_receipt.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match"):
        trainer._persist_latent_scale_state(tmp_path)


def test_checkpoint_latent_scale_state_must_match_current_fit() -> None:
    trainer = EpiForecasterTrainer.__new__(EpiForecasterTrainer)
    trainer.latent_scale_state = _latent_scale_state()
    checkpoint = {
        "latent_scale_state": trainer.latent_scale_state.to_dict(),
        "latent_scale_state_hash": trainer.latent_scale_state.state_hash,
    }

    trainer._validate_checkpoint_latent_scale_state(checkpoint)

    checkpoint["latent_scale_state_hash"] = "wrong"
    with pytest.raises(ValueError, match="does not match"):
        trainer._validate_checkpoint_latent_scale_state(checkpoint)


def test_persist_latent_gradient_probe_records_sorted_parameter_identity(
    tmp_path,
) -> None:
    trainer = EpiForecasterTrainer.__new__(EpiForecasterTrainer)
    trainer._latent_gradient_probe_names = ["rate.z", "initial_states.a"]

    trainer._persist_latent_gradient_probe(tmp_path)

    payload = json.loads(
        (tmp_path / "latent_gradient_probe.json").read_text(encoding="utf-8")
    )
    assert payload["parameter_names"] == ["initial_states.a", "rate.z"]
    assert len(payload["parameter_name_hash"]) == 64


def test_compute_mean_target_mae_averages_finite_available_targets() -> None:
    metrics = {
        "mae_ww_log1p_per_100k": 1.0,
        "mae_hosp_log1p_per_100k": 2.0,
        "mae_cases_log1p_per_100k": float("nan"),
        "mae_deaths_log1p_per_100k": None,
        "rmse_ww_log1p_per_100k": 100.0,
    }

    assert compute_mean_target_mae(metrics) == pytest.approx(1.5)


def test_compute_mean_target_mae_returns_inf_when_no_finite_targets() -> None:
    metrics = {
        "mae_ww_log1p_per_100k": float("nan"),
        "mae_hosp_log1p_per_100k": float("inf"),
    }

    assert compute_mean_target_mae(metrics) == float("inf")


def test_compute_mean_target_crps_averages_finite_available_targets() -> None:
    metrics = {
        "crps_ww_log1p_per_100k": 0.5,
        "crps_hosp_log1p_per_100k": 1.5,
        "crps_cases_log1p_per_100k": float("nan"),
        "crps_deaths_log1p_per_100k": None,
        "mae_hosp_log1p_per_100k": 100.0,
    }

    assert compute_mean_target_crps(metrics) == pytest.approx(1.0)


def test_compute_mean_target_nll_averages_finite_available_targets() -> None:
    metrics = {
        "nll_ww_log1p_per_100k": -0.5,
        "nll_hosp_log1p_per_100k": 1.5,
        "nll_cases_log1p_per_100k": float("nan"),
        "nll_deaths_log1p_per_100k": None,
    }

    assert compute_mean_target_nll(metrics) == pytest.approx(0.5)


def test_compute_validation_selection_score_prefers_crps_for_distributional_loss() -> (
    None
):
    metrics = {
        "mae_ww_log1p_per_100k": 10.0,
        "mae_hosp_log1p_per_100k": 12.0,
        "crps_ww_log1p_per_100k": 0.5,
        "crps_hosp_log1p_per_100k": 1.5,
    }

    score, metric_name = compute_validation_selection_score(
        metrics,
        observation_loss="nll",
    )

    assert score == pytest.approx(1.0)
    assert metric_name == "mean_target_crps"


def test_compute_validation_selection_score_keeps_mae_for_point_loss() -> None:
    metrics = {
        "mae_ww_log1p_per_100k": 10.0,
        "mae_hosp_log1p_per_100k": 12.0,
        "crps_ww_log1p_per_100k": 0.5,
        "crps_hosp_log1p_per_100k": 1.5,
    }

    score, metric_name = compute_validation_selection_score(
        metrics,
        observation_loss="mse",
    )

    assert score == pytest.approx(11.0)
    assert metric_name == "mean_target_mae"


def test_compute_validation_selection_score_uses_nll_when_crps_unavailable() -> None:
    metrics = {
        "mae_ww_log1p_per_100k": 10.0,
        "nll_ww_log1p_per_100k": -0.5,
        "nll_hosp_log1p_per_100k": 1.5,
        "crps_ww_log1p_per_100k": float("nan"),
    }

    score, metric_name = compute_validation_selection_score(
        metrics,
        observation_loss="nll",
    )

    assert score == pytest.approx(0.5)
    assert metric_name == "mean_target_nll"


def test_normalize_training_history_migrates_legacy_selection_series() -> None:
    history = normalize_training_history(
        {
            "train_loss": [1.0],
            "val_mae": [0.4],
        },
        default_selection_metric="mean_target_mae",
    )

    assert "val_mae" not in history
    assert history["val_selection_score"] == [0.4]
    assert history["val_selection_metric"] == ["mean_target_mae"]


def test_run_reports_selection_score_to_optuna_and_tracks_joint_loss() -> None:
    trainer = EpiForecasterTrainer.__new__(EpiForecasterTrainer)
    reports: list[tuple[float, int]] = []

    class _Trial:
        def report(self, value: float, step: int) -> None:
            reports.append((value, step))

        def should_prune(self) -> bool:
            return False

    class _Model:
        def parameters(self):
            return []

    trainer.config = SimpleNamespace(
        output=SimpleNamespace(
            experiment_name="selection_test",
            save_checkpoints=True,
            checkpoint_frequency=1,
            save_best_only=True,
            write_granular_eval=False,
        ),
        training=SimpleNamespace(
            profiler=SimpleNamespace(enabled=False),
            max_batches=None,
            epochs=1,
            scheduler_type="none",
            plot_forecasts=False,
            run_final_evaluation=False,
        ),
    )
    trainer.experiment_dir = None
    trainer.wandb_run = None
    trainer.current_epoch = 0
    trainer.global_step = 0
    trainer.device = torch.device("cpu")
    trainer.nan_loss_triggered = False
    trainer.best_val_loss = float("inf")
    trainer.best_val_selection_score = float("inf")
    trainer.best_val_selection_metric = "mean_target_mae"
    trainer.best_val_joint_loss = float("inf")
    trainer.best_val_series_metrics = {}
    trainer.patience_counter = 0
    trainer.scheduler = None
    trainer.criterion = SimpleNamespace(observation_loss="mse")
    trainer.trial = _Trial()
    trainer.pruning_start_epoch = 0
    trainer.val_loader = object()
    trainer.test_loader = object()
    trainer.model = _Model()
    trainer._status = lambda *args, **kwargs: None
    trainer._train_epoch = lambda: 123.0
    trainer._log_epoch = lambda *args, **kwargs: None
    trainer._write_main_model_aggregate_csvs = lambda *args, **kwargs: None
    trainer._shutdown_loader_iterator = lambda *args, **kwargs: None
    trainer._is_early_stopping_enabled = lambda: True
    trainer._save_checkpoint = lambda *args, **kwargs: None
    trainer._best_checkpoint_path = lambda: Path("best_model.pt")
    trainer._load_best_model_for_final_evaluation = lambda: pytest.fail(
        "workflow-managed training must not run final evaluation"
    )
    trainer.cleanup_dataloaders = lambda: None
    trainer._evaluate_split = lambda *args, **kwargs: (
        999.0,
        {
            "mae_ww_log1p_per_100k": 1.0,
            "mae_hosp_log1p_per_100k": 3.0,
            "mae_cases_log1p_per_100k": float("nan"),
            "mae_deaths_log1p_per_100k": None,
            "loss_ww": 10.0,
        },
        {},
    )
    trainer.test_epoch = lambda **_kwargs: pytest.fail(
        "workflow-managed training must not run the test split"
    )

    results = trainer.run()

    assert reports == [(2.0, 0)]
    assert results["best_val_selection_score"] == pytest.approx(2.0)
    assert results["best_val_loss"] == pytest.approx(2.0)
    assert results["best_val_joint_loss"] == pytest.approx(999.0)
    assert results["best_val_series_metrics"]["loss_ww"] == pytest.approx(10.0)
    assert trainer.training_history["val_selection_score"] == [2.0]
    assert trainer.training_history["val_selection_metric"] == ["mean_target_mae"]
    assert "val_mae" not in trainer.training_history


def test_best_checkpoint_path_raises_when_best_model_missing(tmp_path) -> None:
    trainer = EpiForecasterTrainer.__new__(EpiForecasterTrainer)
    trainer.config = SimpleNamespace(output=SimpleNamespace(save_checkpoints=True))
    trainer.checkpoint_dir = tmp_path / "checkpoints"
    trainer.checkpoint_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="best checkpoint is missing"):
        trainer._best_checkpoint_path()


class TestEpiForecasterTrainer:
    @pytest.fixture
    def minimal_config(self, tmp_path):
        return EpiForecasterConfig(
            model=ModelConfig(
                type=ModelVariant(cases=True),
                mobility_embedding_dim=8,
                region_embedding_dim=8,
                input_window_length=14,
                forecast_horizon=7,
                max_neighbors=5,
                sir_physics=SIRPhysicsConfig(),
                observation_heads=ObservationHeadConfig(),
            ),
            data=DataConfig(
                dataset_path="dummy_dataset.zarr",
                run_id="real",
                run_id_chunk_size=1,
            ),
            training=TrainingParams(
                epochs=1,
                batch_size=4,
                loss=LossConfig(name="joint_inference"),
                curriculum=CurriculumConfig(enabled=False),
                enable_mixed_precision=False,
            ),
            output=OutputConfig(log_dir=str(tmp_path), experiment_name="test_exp"),
        )

    def test_early_stopping_disabled_during_synth_only_curriculum(self):
        trainer = EpiForecasterTrainer.__new__(EpiForecasterTrainer)
        trainer.config = SimpleNamespace(
            training=SimpleNamespace(early_stopping_patience=5),
        )
        trainer.real_run_id = "real"
        trainer.curriculum_sampler = SimpleNamespace(
            state=SimpleNamespace(synth_ratio=1.0)
        )

        assert trainer._is_early_stopping_enabled() is False

    def test_early_stopping_enabled_once_curriculum_mixes_real_data(self):
        trainer = EpiForecasterTrainer.__new__(EpiForecasterTrainer)
        trainer.config = SimpleNamespace(
            training=SimpleNamespace(early_stopping_patience=5),
        )
        trainer.real_run_id = "real"
        trainer.curriculum_sampler = SimpleNamespace(
            state=SimpleNamespace(synth_ratio=0.8)
        )

        assert trainer._is_early_stopping_enabled() is True

    @patch("data.dataset_factory.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    @patch("training.dataloader_factory.DataLoader")
    def test_standard_split_reuses_sparse_topology_and_releases(
        self,
        mock_dataloader,
        mock_wandb,
        mock_model_cls,
        mock_dataset_cls,
        minimal_config,
    ):
        """Val/test should reuse train full sparse topology and release references."""
        mock_model_instance = MagicMock()
        param = torch.nn.Parameter(torch.randn(1))
        mock_model_instance.parameters.return_value = [param]
        mock_model_instance.named_parameters.return_value = [("backbone.mock", param)]
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        shared_mobility = object()
        shared_mobility_mask = object()
        shared_sparse_topology = object()

        train_ds = MagicMock()
        train_ds.cases_output_dim = 1
        train_ds.biomarkers_output_dim = 1
        train_ds.temporal_covariates_dim = 0
        train_ds.target_nodes = [0, 1]
        train_ds.__len__.return_value = 16
        train_ds.preloaded_mobility = shared_mobility
        train_ds.mobility_mask = shared_mobility_mask
        train_ds.shared_sparse_topology = shared_sparse_topology
        train_ds.biomarker_preprocessor = object()
        train_ds.mobility_preprocessor = object()
        train_ds.region_embeddings = None

        val_ds = MagicMock()
        val_ds.target_nodes = [2]
        val_ds.__len__.return_value = 8
        val_ds.cases_output_dim = 1
        val_ds.biomarkers_output_dim = 1

        test_ds = MagicMock()
        test_ds.target_nodes = [3]
        test_ds.__len__.return_value = 8
        test_ds.cases_output_dim = 1
        test_ds.biomarkers_output_dim = 1

        mock_dataset_cls.side_effect = [train_ds, val_ds, test_ds]

        with patch(
            "data.dataset_factory.split_nodes_by_ratio",
            return_value=([0, 1], [2], [3]),
        ):
            _ = EpiForecasterTrainer(minimal_config)

        val_call = mock_dataset_cls.call_args_list[1]
        test_call = mock_dataset_cls.call_args_list[2]
        assert val_call.kwargs["shared_sparse_topology"] is shared_sparse_topology
        assert test_call.kwargs["shared_sparse_topology"] is shared_sparse_topology
        assert val_call.kwargs["preloaded_mobility"] is shared_mobility
        assert test_call.kwargs["preloaded_mobility"] is shared_mobility
        train_ds.release_shared_sparse_topology.assert_called_once()
        val_ds.release_shared_sparse_topology.assert_called_once()
        test_ds.release_shared_sparse_topology.assert_called_once()

    @patch("data.dataset_factory.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    @patch("training.dataloader_factory.DataLoader")
    def test_gradnorm_compiles_adaptive_backward_step(
        self, mock_dataloader, mock_wandb, mock_model_cls, mock_dataset, minimal_config
    ):
        mock_model_instance = MagicMock()
        param = torch.nn.Parameter(torch.randn(1))
        mock_model_instance.parameters.return_value = [param]
        mock_model_instance.named_parameters.return_value = [("backbone.mock", param)]
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.cases_output_dim = 1
        mock_dataset_instance.biomarkers_output_dim = 1
        mock_dataset_instance.temporal_covariates_dim = 0
        mock_dataset_instance.__len__.return_value = 16
        mock_dataset_instance.target_nodes = [0, 1]
        mock_dataset.return_value = mock_dataset_instance
        _mock_canonical_region_coordinate(mock_dataset)

        minimal_config.training.compile_backward = True
        minimal_config.training.loss.joint.observation_loss = "crps"
        minimal_config.training.loss.joint.adaptive_scheme = "gradnorm"
        trainer = EpiForecasterTrainer(minimal_config)
        assert trainer._compiled_training_step is not None
        assert trainer.gradnorm_controller is not None
        assert trainer.gradnorm_optimizer is not None

    @patch("data.dataset_factory.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    @patch("training.dataloader_factory.DataLoader")
    def test_checkpoint_includes_gradnorm_state(
        self,
        mock_dataloader,
        mock_wandb,
        mock_model_cls,
        mock_dataset,
        minimal_config,
        tmp_path,
    ):
        mock_model_instance = MagicMock()
        param = torch.nn.Parameter(torch.randn(1))
        mock_model_instance.parameters.return_value = [param]
        mock_model_instance.named_parameters.return_value = [("backbone.mock", param)]
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.state_dict.return_value = {"backbone.mock": param.detach()}
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.cases_output_dim = 1
        mock_dataset_instance.biomarkers_output_dim = 1
        mock_dataset_instance.temporal_covariates_dim = 0
        mock_dataset_instance.__len__.return_value = 16
        mock_dataset_instance.target_nodes = [0, 1]
        mock_dataset.return_value = mock_dataset_instance
        _mock_canonical_region_coordinate(mock_dataset)

        minimal_config.output.log_dir = str(tmp_path)
        minimal_config.output.experiment_name = "gradnorm_ckpt"
        minimal_config.training.loss.joint.observation_loss = "crps"
        minimal_config.training.loss.joint.adaptive_scheme = "gradnorm"
        trainer = EpiForecasterTrainer(minimal_config)
        trainer._save_checkpoint(epoch=0, val_loss=1.0, is_best=False, is_final=False)

        ckpt_files = list((trainer.checkpoint_dir).glob("checkpoint_epoch_*.pt"))
        assert ckpt_files
        checkpoint = torch.load(ckpt_files[0], map_location="cpu", weights_only=False)
        assert "gradnorm_controller_state_dict" in checkpoint
        assert "gradnorm_optimizer_state_dict" in checkpoint
        assert checkpoint["val_selection_metric"] == "mean_target_mae"



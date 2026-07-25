from pathlib import Path

import yaml

from models.configs import EpiForecasterConfig


def test_synthetic_pretraining_keeps_fractional_mobility_edges() -> None:
    config = EpiForecasterConfig.load(
        Path("configs/production_only/train_epifor_mn5_synth_pretrain.yaml")
    )

    assert config.data.mobility_min_flow_raw == 0.0
    assert config.model.max_neighbors == 20


def test_synthetic_pretrain_interactive_smoke_disables_all_compilation() -> None:
    config = yaml.safe_load(
        Path(
            "workflow/config/mn5_synth_pretrain_interactive_smoke.yaml"
        ).read_text()
    )

    overrides = set(config["training"]["overrides"])
    assert "training.compile=false" in overrides
    assert "training.compile_backward=false" in overrides
    assert "training.pathway_gradient_frequency=1" in overrides
    assert config["workflow"]["target"] == "train_evaluate"
    assert config["evaluation"]["granular"] is True


def test_synthetic_pretrain_production_emits_evaluation_diagnostics() -> None:
    config = yaml.safe_load(
        Path("workflow/config/mn5_synth_pretrain.yaml").read_text()
    )
    training = EpiForecasterConfig.load(
        Path("configs/production_only/train_epifor_mn5_synth_pretrain.yaml")
    )

    assert config["workflow"]["target"] == "train_evaluate"
    assert config["evaluation"]["granular"] is True
    assert training.training.pathway_gradient_frequency == 500


def test_synthetic_pretrain_control_matches_real_training_contract() -> None:
    control = yaml.safe_load(
        Path("workflow/config/mn5_synth_pretrain_control.yaml").read_text()
    )
    canonical = yaml.safe_load(Path("workflow/config/mn5_train.yaml").read_text())

    assert control["workflow"]["target"] == "train_evaluate"
    assert control["training"]["base_config"] == canonical["training"]["base_config"]
    assert control["training"]["max_batches"] is None
    assert control["training"]["overrides"] == canonical["training"]["overrides"]
    assert control["dvc"] == canonical["dvc"]
    assert control["evaluation"] == canonical["evaluation"]
    assert control["resources"] == canonical["resources"]

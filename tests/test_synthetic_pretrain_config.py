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

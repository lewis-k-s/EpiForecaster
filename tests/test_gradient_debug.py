"""Tests for gradient_debug utilities."""

import json

import pytest
import torch
import torch.nn as nn

from utils.gradient_debug import (
    GradientDebugger,
    GradientSnapshot,
    GradientStats,
    create_gradient_debugger,
    has_non_finite_gradients,
)


class SimpleModel(nn.Module):
    """Simple model for testing gradient diagnostics."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class HeadModel(nn.Module):
    """Small model with observation-head style names for snapshot tests."""

    def __init__(self):
        super().__init__()
        self.ww_head = nn.Linear(4, 2)
        self.deaths_head = nn.Linear(2, 1)

    def forward(self, x):
        return self.deaths_head(torch.relu(self.ww_head(x)))


class TestGradientDebugger:
    """Test suite for GradientDebugger."""

    def test_disabled_debugger_has_no_effect(self):
        """Disabled debugger should be no-op."""
        debugger = GradientDebugger(enabled=False)
        model = SimpleModel()
        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        # These should not raise or modify anything
        assert not debugger.check_gradients(model)
        snapshot = debugger.capture_snapshot(model)
        assert snapshot.step == 0
        assert not snapshot.has_non_finite

    def test_enabled_debugger_detects_non_finite_gradients(self, tmp_path):
        """Debugger should detect non-finite gradients when enabled."""
        debugger = GradientDebugger(enabled=True, log_dir=tmp_path)
        model = SimpleModel()

        # Inject non-finite gradient
        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()
        model.fc1.weight.grad[0, 0] = float("nan")

        assert debugger.check_gradients(model)

    def test_capture_snapshot_produces_correct_structure(self, tmp_path):
        """Snapshot should have correct structure and data."""
        debugger = GradientDebugger(enabled=True, log_dir=tmp_path)
        model = SimpleModel()

        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        step_info = {"step": 42, "epoch": 3, "batch_idx": 7}
        snapshot = debugger.capture_snapshot(model, loss=y, step_info=step_info)

        assert snapshot.step == 42
        assert snapshot.epoch == 3
        assert snapshot.batch_idx == 7
        assert snapshot.loss == pytest.approx(y.item())
        assert len(snapshot.layer_stats) == 4  # 2 weights + 2 biases
        assert not snapshot.has_non_finite

    def test_snapshot_detects_non_finite_layers(self, tmp_path):
        """Snapshot should correctly identify layers with non-finite gradients."""
        debugger = GradientDebugger(enabled=True, log_dir=tmp_path)
        model = SimpleModel()

        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        # Inject NaN
        model.fc1.weight.grad[0, 0] = float("nan")

        snapshot = debugger.capture_snapshot(model)

        assert snapshot.has_non_finite
        assert "fc1.weight" in snapshot.non_finite_layers
        assert len(snapshot.non_finite_layers) == 1

    def test_save_report_creates_valid_json(self, tmp_path):
        """Save report should create valid JSON file."""
        debugger = GradientDebugger(enabled=True, log_dir=tmp_path)
        model = SimpleModel()

        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        snapshot = debugger.capture_snapshot(model, step_info={"step": 123})
        filepath = debugger.save_report(snapshot)

        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)

        assert data["step"] == 123
        assert "layer_stats" in data
        assert "summary" in data

    def test_layer_stats_computed_correctly(self, tmp_path):
        """Layer statistics should be accurate."""
        debugger = GradientDebugger(enabled=True, log_dir=tmp_path)
        model = SimpleModel()

        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        # Inject specific pattern
        model.fc1.weight.grad[0, 0] = float("nan")
        model.fc1.weight.grad[0, 1] = float("inf")

        stats = debugger.compute_layer_stats("fc1.weight", model.fc1.weight)

        assert stats is not None
        assert stats.name == "fc1.weight"
        assert stats.nan_count == 1
        assert stats.inf_count == 1
        assert stats.finite_ratio < 1.0
        assert stats.shape == [5, 10]
        assert stats.numel == 50

    def test_summary_generated_correctly(self, tmp_path):
        """Summary should aggregate statistics correctly."""
        debugger = GradientDebugger(enabled=True, log_dir=tmp_path)
        model = SimpleModel()

        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        # Inject issues in multiple layers
        model.fc1.weight.grad[0, 0] = float("nan")
        model.fc2.weight.grad[0, 0] = float("inf")

        snapshot = debugger.capture_snapshot(model)

        assert snapshot.summary["total_nan_values"] == 1
        assert snapshot.summary["total_inf_values"] == 1
        assert snapshot.summary["non_finite_layers_count"] == 2
        assert len(snapshot.summary["most_problematic_layers"]) == 2

    def test_snapshot_tracks_vanishing_and_exploding_layers(self, tmp_path):
        """Snapshot summary should flag very small and very large layer norms."""
        debugger = GradientDebugger(
            enabled=True,
            log_dir=tmp_path,
            vanishing_threshold=1.0e-6,
            exploding_threshold=10.0,
            snapshot_top_k=2,
        )
        model = SimpleModel()

        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        model.fc1.bias.grad.zero_()
        model.fc2.weight.grad.fill_(50.0)

        snapshot = debugger.capture_snapshot(model, step_info={"step": 9})

        assert "fc1.bias" in snapshot.vanishing_layers
        assert "fc2.weight" in snapshot.exploding_layers
        assert snapshot.summary["vanishing_layers_count"] == 1
        assert snapshot.summary["exploding_layers_count"] == 1
        assert snapshot.summary["highest_norm_layers"][0]["name"] == "fc2.weight"
        assert snapshot.summary["lowest_norm_layers"][0]["name"] == "fc1.bias"

    def test_snapshot_log_data_and_status_are_compact_and_numeric(self, tmp_path):
        """Snapshot helpers should expose metrics for logging surfaces."""
        debugger = GradientDebugger(enabled=True, log_dir=tmp_path)
        model = SimpleModel()

        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        snapshot = debugger.capture_snapshot(model, step_info={"step": 11})
        log_data = debugger.build_snapshot_log_data(snapshot)
        status = debugger.format_snapshot_status(snapshot)

        assert log_data["grad_snapshot_layers_with_grads"] == 4
        assert "grad_snapshot_max_layer_norm" in log_data
        assert "Gradient snapshot @ step 11" in status
        assert "vanishing=" in status
        assert "exploding=" in status

    def test_snapshot_marks_expected_vs_unexpected_zero_heads(self, tmp_path):
        """Aggregate head health should not fail active heads for one dead parameter."""
        debugger = GradientDebugger(enabled=True, log_dir=tmp_path)
        model = HeadModel()

        x = torch.randn(2, 4)
        y = model(x).sum()
        y.backward()

        model.ww_head.weight.grad.zero_()
        model.ww_head.bias.grad.zero_()
        model.deaths_head.bias.grad.zero_()

        snapshot = debugger.capture_snapshot(
            model,
            step_info={"step": 21},
            head_supervision={
                "ww": {
                    "active": False,
                    "n_eff": 0.0,
                    "valid_points": 0,
                    "valid_series": 0,
                },
                "deaths": {
                    "active": True,
                    "n_eff": 6.0,
                    "valid_points": 6,
                    "valid_series": 2,
                },
            },
            head_coverage={
                "ww": {"pass_rate": 0.2, "zero_when_active_rate": 0.0, "zero_when_inactive_rate": 1.0},
                "deaths": {"pass_rate": 0.8, "zero_when_active_rate": 0.0, "zero_when_inactive_rate": 0.0},
            },
        )

        ww_health = snapshot.head_gradient_health["ww"]
        deaths_health = snapshot.head_gradient_health["deaths"]
        assert ww_health["expected_zero"]
        assert not ww_health["unexpected_zero"]
        assert ww_health["grad_norm"] == pytest.approx(0.0)
        assert deaths_health["has_vanishing_layers"]
        assert deaths_health["vanishing_layer_count"] == 1
        assert deaths_health["grad_norm"] > debugger.vanishing_threshold
        assert not deaths_health["unexpected_zero"]

        log_data = debugger.build_snapshot_log_data(snapshot)
        assert log_data["grad_snapshot_head_ww_active"] == 0
        assert log_data["grad_snapshot_head_ww_grad_norm"] == pytest.approx(0.0)
        assert log_data["grad_snapshot_head_ww_expected_zero"] == 1
        assert log_data["grad_snapshot_head_deaths_grad_norm"] > 0.0
        assert log_data["grad_snapshot_head_ww_pass_rate"] == pytest.approx(0.2)
        assert "expected-zero" in debugger.format_snapshot_status(snapshot)


class TestHelperFunctions:
    """Test utility helper functions."""

    def test_has_non_finite_gradients_detects_nan(self):
        """Quick check should detect NaN gradients."""
        model = SimpleModel()
        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        # Initially all finite
        assert not has_non_finite_gradients(model)

        # Inject NaN
        model.fc1.weight.grad[0, 0] = float("nan")
        assert has_non_finite_gradients(model)

    def test_has_non_finite_gradients_detects_inf(self):
        """Quick check should detect Inf gradients."""
        model = SimpleModel()
        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        # Inject Inf
        model.fc1.weight.grad[0, 0] = float("inf")
        assert has_non_finite_gradients(model)

    def test_create_gradient_debugger_from_config(self, tmp_path):
        """Factory should create debugger from config dict."""
        from types import SimpleNamespace

        config = SimpleNamespace(
            enable_gradient_debug=True,
            gradient_debug_log_dir=str(tmp_path),
            gradient_vanishing_threshold=1.0e-7,
            gradient_exploding_threshold=50.0,
            gradient_snapshot_top_k=3,
        )

        debugger = create_gradient_debugger(config)
        assert debugger.enabled
        assert debugger.log_dir == tmp_path
        assert debugger.vanishing_threshold == pytest.approx(1.0e-7)
        assert debugger.exploding_threshold == pytest.approx(50.0)
        assert debugger.snapshot_top_k == 3

    def test_create_gradient_debugger_disabled_by_default(self):
        """Factory should create disabled debugger with None config."""
        debugger = create_gradient_debugger(None)
        assert not debugger.enabled


class TestGradientDataClasses:
    """Test GradientStats and GradientSnapshot dataclasses."""

    def test_gradient_stats_defaults(self):
        """GradientStats should have sensible defaults."""
        stats = GradientStats(name="test")
        assert stats.name == "test"
        assert stats.numel == 0
        assert stats.finite_ratio == 1.0
        assert stats.nan_count == 0
        assert stats.inf_count == 0

    def test_gradient_snapshot_defaults(self):
        """GradientSnapshot should have sensible defaults."""
        snapshot = GradientSnapshot(step=1, epoch=0, batch_idx=0)
        assert snapshot.step == 1
        assert not snapshot.has_non_finite
        assert snapshot.layer_stats == []
        assert snapshot.non_finite_layers == []
        assert snapshot.vanishing_layers == []
        assert snapshot.exploding_layers == []

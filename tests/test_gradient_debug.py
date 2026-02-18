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
        )

        debugger = create_gradient_debugger(config)
        assert debugger.enabled
        assert debugger.log_dir == tmp_path

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

"""Tests for sparsity-loss correlation logging utilities."""

import pytest
import torch


class TestComputeBatchSparsity:
    """Tests for compute_batch_sparsity function."""

    def test_clinical_history_sparsity(self):
        """Test sparsity computation for 3-channel clinical series."""
        from utils.sparsity_logging import compute_batch_sparsity

        B, L = 4, 28
        batch = {
            "B": B,
            "HospHist": torch.zeros(B, L, 3),
            "DeathsHist": torch.zeros(B, L, 3),
            "CasesHist": torch.zeros(B, L, 3),
        }

        # Set masks: sample 0 fully observed, sample 1 half observed, sample 2 25% observed, sample 3 fully missing
        for key in ["HospHist", "DeathsHist", "CasesHist"]:
            batch[key][0, :, 1] = 1.0  # 0% sparsity
            batch[key][1, : L // 2, 1] = 1.0  # 50% sparsity
            batch[key][2, : L // 4, 1] = 1.0  # 75% sparsity
            # sample 3: all zeros = 100% sparsity

        sparsity = compute_batch_sparsity(batch)

        assert "hosp_hist" in sparsity
        assert "deaths_hist" in sparsity
        assert "cases_hist" in sparsity

        # Check sparsity values
        hosp = sparsity["hosp_hist"]
        assert torch.isclose(hosp[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(hosp[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(hosp[2], torch.tensor(0.75), atol=1e-5)
        assert torch.isclose(hosp[3], torch.tensor(1.0), atol=1e-5)

    def test_target_sparsity(self):
        """Test sparsity computation for forecast horizon targets."""
        from utils.sparsity_logging import compute_batch_sparsity

        B, H = 4, 14
        batch = {
            "B": B,
            "HospTargetMask": torch.zeros(B, H),
            "WWTargetMask": torch.zeros(B, H),
        }

        # Sample 0: fully observed, sample 1: half, sample 2: quarter, sample 3: none
        batch["HospTargetMask"][0, :] = 1.0
        batch["HospTargetMask"][1, : H // 2] = 1.0
        batch["HospTargetMask"][2, : H // 4] = 1.0

        batch["WWTargetMask"][0, :] = 1.0
        batch["WWTargetMask"][1, : H // 2] = 1.0

        sparsity = compute_batch_sparsity(batch)

        assert "hosp_target" in sparsity
        assert "ww_target" in sparsity

        hosp = sparsity["hosp_target"]
        assert torch.isclose(hosp[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(hosp[1], torch.tensor(0.5), atol=1e-5)
        # 14 // 4 = 3, so 11/14 = 0.7857 (integer division rounding)
        assert torch.isclose(hosp[2], torch.tensor(11 / 14), atol=1e-5)
        assert torch.isclose(hosp[3], torch.tensor(1.0), atol=1e-5)

    def test_empty_batch_returns_empty_dict(self):
        """Test that empty batch returns empty sparsity dict."""
        from utils.sparsity_logging import compute_batch_sparsity

        batch = {"B": 1}
        sparsity = compute_batch_sparsity(batch)

        # Should not crash, may have empty or default values
        assert isinstance(sparsity, dict)


class TestComputePerSampleHeadLosses:
    """Tests for compute_per_sample_head_losses function."""

    def test_per_sample_mse_loss(self):
        """Test per-sample MSE loss computation."""
        from utils.sparsity_logging import compute_per_sample_head_losses

        B, H = 4, 14

        model_outputs = {
            "pred_hosp": torch.randn(B, H),
            "pred_ww": torch.randn(B, H),
        }

        targets = {
            "hosp": torch.randn(B, H),
            "hosp_mask": torch.ones(B, H),
            "ww": torch.randn(B, H),
            "ww_mask": torch.ones(B, H),
        }

        # Make some targets NaN to test masking
        targets["hosp"][2, :] = float("nan")
        targets["ww"][3, 5:10] = float("nan")

        losses = compute_per_sample_head_losses(model_outputs, targets)

        assert "loss_hosp" in losses
        assert "loss_ww" in losses

        # Check shape
        assert losses["loss_hosp"].shape == (B,)
        assert losses["loss_ww"].shape == (B,)

        # Check finite values
        assert torch.isfinite(losses["loss_hosp"]).all()
        assert torch.isfinite(losses["loss_ww"]).all()

    def test_per_sample_loss_with_missing_mask(self):
        """Test that missing mask defaults to all-ones."""
        from utils.sparsity_logging import compute_per_sample_head_losses

        B, H = 2, 7

        model_outputs = {
            "pred_hosp": torch.ones(B, H),
        }

        targets = {
            "hosp": torch.zeros(B, H),
        }

        losses = compute_per_sample_head_losses(model_outputs, targets)

        assert "loss_hosp" in losses
        # Loss should be 1.0 (MSE of ones vs zeros)
        assert torch.isclose(losses["loss_hosp"], torch.ones(B)).all()


class TestLogSparsityLossCorrelation:
    """Tests for log_sparsity_loss_correlation function."""

    def test_skips_when_wandb_is_none(self, capsys):
        """Test that function silently skips when wandb_run is None."""
        from utils.sparsity_logging import log_sparsity_loss_correlation

        batch = {"B": 2, "HospHist": torch.zeros(2, 28, 3)}
        model_outputs = {"pred_hosp": torch.zeros(2, 14)}
        targets = {"hosp": torch.zeros(2, 14), "hosp_mask": torch.ones(2, 14)}

        log_sparsity_loss_correlation(
            batch=batch,
            model_outputs=model_outputs,
            targets=targets,
            wandb_run=None,
            step=0,
        )

        captured = capsys.readouterr()
        assert len(captured.out) == 0

    def test_logs_summary_stats_when_wandb_available(self, monkeypatch):
        """Test that W&B summary statistics are logged correctly."""
        from utils.sparsity_logging import log_sparsity_loss_correlation

        logged_data = {}

        class MockWandbRun:
            pass

        def mock_log(data, step=None):
            logged_data.update(data)
            logged_data["_step"] = step

        import sys
        import types

        mock_wandb = types.ModuleType("wandb")
        mock_wandb.log = mock_log
        sys.modules["wandb"] = mock_wandb

        import importlib

        import utils.sparsity_logging as sl_module

        importlib.reload(sl_module)

        B, L, H = 2, 28, 14
        batch = {
            "B": B,
            "HospHist": torch.zeros(B, L, 3),
            "BioNode": torch.zeros(B, L, 13),
        }
        batch["HospHist"][0, :, 1] = 1.0
        batch["HospHist"][1, : L // 2, 1] = 1.0
        batch["BioNode"][:, :, 1] = 1.0

        model_outputs = {
            "pred_hosp": torch.ones(B, H),
            "pred_ww": torch.ones(B, H),
        }
        targets = {
            "hosp": torch.zeros(B, H),
            "hosp_mask": torch.ones(B, H),
            "ww": torch.zeros(B, H),
            "ww_mask": torch.ones(B, H),
        }

        log_sparsity_loss_correlation(
            batch=batch,
            model_outputs=model_outputs,
            targets=targets,
            wandb_run=MockWandbRun(),
            step=42,
            epoch=1,
        )

        # Check that summary statistics are logged for each head
        assert "sparsity_loss_hosp_mean" in logged_data
        assert "sparsity_loss_hosp_std" in logged_data
        assert "sparsity_loss_hosp_p25" in logged_data
        assert "sparsity_loss_hosp_p50" in logged_data
        assert "sparsity_loss_hosp_p75" in logged_data
        assert "sparsity_loss_hosp_max" in logged_data
        assert "sparsity_loss_ww_mean" in logged_data
        assert logged_data["_step"] == 42

        # Verify values are reasonable (loss=1.0, sparsity varies between 0-0.5)
        # Sample 1: sparsity = 0 (fully observed), loss = 1.0, product = 0.0
        # Sample 2: sparsity = 0.5 (half observed), loss = 1.0, product = 0.5
        # Mean should be 0.25
        assert logged_data["sparsity_loss_hosp_mean"] == 0.25
        assert logged_data["sparsity_loss_hosp_max"] == 0.5

    @pytest.mark.device
    def test_cross_device_sparsity_loss_correlation(self, accelerator_device):
        """Test that sparsity-loss correlation works when batch is CPU and losses are on accelerator.

        This simulates the real training scenario where:
        - Batch data comes from DataLoader on CPU
        - Model outputs and targets are on GPU/MPS after forward pass

        Regression test for device mismatch bug where sparsity_vals (CPU) was
        multiplied by loss_vals (GPU) without device synchronization.
        """
        from utils.sparsity_logging import log_sparsity_loss_correlation

        logged_data = {}

        class MockWandbRun:
            pass

        def mock_log(data, step=None):
            logged_data.update(data)

        import sys
        import types

        mock_wandb = types.ModuleType("wandb")
        mock_wandb.log = mock_log
        sys.modules["wandb"] = mock_wandb

        import importlib

        import utils.sparsity_logging as sl_module

        importlib.reload(sl_module)

        B, L, H = 2, 28, 14

        # Batch data on CPU (simulates DataLoader output)
        batch = {
            "B": B,
            "HospHist": torch.zeros(B, L, 3),  # CPU
            "BioNode": torch.zeros(B, L, 13),  # CPU
        }
        batch["HospHist"][:, :, 1] = 1.0  # mask channel: fully observed
        batch["BioNode"][:, :, 1] = 1.0

        # Model outputs and targets on accelerator (simulates GPU forward pass)
        model_outputs = {
            "pred_hosp": torch.ones(B, H).to(accelerator_device),
            "pred_ww": torch.ones(B, H).to(accelerator_device),
        }
        targets = {
            "hosp": torch.zeros(B, H).to(accelerator_device),
            "hosp_mask": torch.ones(B, H).to(accelerator_device),
            "ww": torch.zeros(B, H).to(accelerator_device),
            "ww_mask": torch.ones(B, H).to(accelerator_device),
        }

        # This should NOT raise RuntimeError about device mismatch
        log_sparsity_loss_correlation(
            batch=batch,
            model_outputs=model_outputs,
            targets=targets,
            wandb_run=MockWandbRun(),
            step=0,
            epoch=1,
        )

        # Verify logging happened
        assert "sparsity_loss_hosp_mean" in logged_data

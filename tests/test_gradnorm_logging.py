from __future__ import annotations

import torch

from utils.gradnorm_logging import did_gradnorm_sidecar_run


def test_did_gradnorm_sidecar_run_truthy_tensor() -> None:
    assert did_gradnorm_sidecar_run({"gradnorm_sidecar_ran": torch.tensor(1.0)})


def test_did_gradnorm_sidecar_run_false_when_missing_or_zero() -> None:
    assert not did_gradnorm_sidecar_run({})
    assert not did_gradnorm_sidecar_run({"gradnorm_sidecar_ran": torch.tensor(0.0)})

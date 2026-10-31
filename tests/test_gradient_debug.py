import json

import torch

from utils.gradient_debug import GradientDebugger


def test_nonfinite_gradient_snapshot_writes_report(tmp_path) -> None:
    model = torch.nn.Linear(3, 1)
    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    assert model.weight.grad is not None
    model.weight.grad[0, 0] = float("nan")

    debugger = GradientDebugger(enabled=True, log_dir=tmp_path)
    assert debugger.check_gradients(model)

    snapshot = debugger.capture_snapshot(
        model,
        loss=loss,
        step_info={"step": 12, "epoch": 2, "batch_idx": 3},
    )
    report_path = debugger.save_report(snapshot)
    report = json.loads(report_path.read_text())

    assert snapshot.has_non_finite
    assert snapshot.non_finite_layers == ["weight"]
    assert report["step"] == 12
    assert report["summary"]["total_nan_values"] == 1
    assert report["summary"]["non_finite_layers_count"] == 1

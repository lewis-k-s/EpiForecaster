"""GPU telemetry tests use fake driver and query tools; no GPU is required."""

import ctypes
import csv
import json
import subprocess
import sys
import time
import uuid

import pytest

from scripts.workflow import gpu_telemetry as telemetry


def test_cuda_ordinals_are_resolved_to_uuids(monkeypatch):
    identifiers = [uuid.uuid4(), uuid.uuid4()]

    class Driver:
        def cuInit(self, flags):
            return 0

        def cuDeviceGetCount(self, count):
            count._obj.value = 2
            return 0

        def cuDeviceGet(self, device, ordinal):
            device._obj.value = ordinal
            return 0

        def cuDeviceGetUuid(self, result, device):
            raise AssertionError("use v2 UUID identity when available")

        def cuDeviceGetUuid_v2(self, result, device):
            ctypes.memmove(result, identifiers[device.value].bytes, 16)
            return 0

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    monkeypatch.setattr(telemetry.ctypes, "CDLL", lambda name: Driver())
    assert telemetry.visible_gpu_uuids() == ["GPU-" + str(x) for x in identifiers]


@pytest.mark.parametrize("visible", ["", "-1", "MIG-example"])
def test_unknown_or_mig_device_selection_fails_closed(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(RuntimeError):
        telemetry.visible_gpu_uuids()


def test_query_is_uuid_scoped_and_preserves_unavailable_values(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 2
        return subprocess.CompletedProcess(
            command, 0, "GPU-a, 32, 20, 512, 64000, N/A, 40\n"
        )

    monkeypatch.setattr(telemetry.subprocess, "run", run)
    rows = telemetry.query_gpus(["GPU-a"], 2)
    assert "--id=GPU-a" in commands[0]
    assert rows[0][1:] == ["GPU-a", "32", "20", "512", "64000", "N/A", "40"]
    with pytest.raises(ValueError, match="identity"):
        telemetry.query_gpus(["GPU-other"], 2)


def test_sampler_stops_and_preserves_each_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(telemetry, "visible_gpu_uuids", lambda: ["GPU-a"])
    monkeypatch.setattr(
        telemetry,
        "query_gpus",
        lambda devices, timeout: [
            [telemetry.now(), "GPU-a", "50", "20", "512", "64000", "100", "40"]
        ],
    )
    (tmp_path / ".state").mkdir()
    (tmp_path / ".state/active_execution").write_text("attempt-1")
    monkeypatch.setenv("EPIFORECASTER_WORKER_ID", "3")
    sampler = telemetry.Sampler(
        tmp_path, telemetry.TelemetryConfig(interval_seconds=0.01)
    )
    sampler.start()
    time.sleep(0.05)
    sampler.close(7)
    assert not sampler.thread.is_alive()
    manifest = json.loads((sampler.root / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["scientific_exit_code"] == 7
    assert manifest["worker_id"] == "3"
    assert manifest["execution_attempt_id"] == "attempt-1"
    assert manifest["samples"] >= 1
    with (sampler.root / "gpu_samples.csv").open() as handle:
        assert next(csv.reader(handle)) == list(telemetry.COLUMNS)
    other = telemetry.Sampler(tmp_path)
    assert other.root != sampler.root


@pytest.mark.parametrize(
    "error", [FileNotFoundError("no driver"), RuntimeError("CUDA failed")]
)
def test_workload_exit_code_survives_telemetry_failure(monkeypatch, tmp_path, error):
    def unavailable():
        raise error

    monkeypatch.setattr(telemetry, "visible_gpu_uuids", unavailable)
    assert (
        telemetry.run_command([sys.executable, "-c", "raise SystemExit(7)"], tmp_path)
        == 7
    )
    manifest = json.loads(
        next(tmp_path.glob("outputs/gpu_telemetry/*/manifest.json")).read_text()
    )
    assert manifest["status"] == "unavailable"
    assert manifest["scientific_exit_code"] == 7


def test_gpu_query_failure_does_not_stop_training(monkeypatch, tmp_path):
    monkeypatch.setattr(telemetry, "visible_gpu_uuids", lambda: ["GPU-a"])

    def failed(devices, timeout):
        raise subprocess.TimeoutExpired("nvidia-smi", timeout)

    monkeypatch.setattr(telemetry, "query_gpus", failed)
    assert (
        telemetry.run_command(
            [sys.executable, "-c", "import time; time.sleep(.05)"], tmp_path
        )
        == 0
    )
    manifest = json.loads(
        next(tmp_path.glob("outputs/gpu_telemetry/*/manifest.json")).read_text()
    )
    assert manifest["status"] == "partial"
    assert manifest["errors"] >= 1


def test_signal_forwarding_preserves_signal_exit_and_closes_sampler(tmp_path):
    import os
    import signal

    ready = tmp_path / "ready"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scripts.workflow.gpu_telemetry",
            "--run-root",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; import time; Path({str(ready)!r}).touch(); time.sleep(30)",
        ]
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        os.kill(process.pid, signal.SIGTERM)
        assert process.wait(timeout=5) == 143
        manifest = json.loads(
            next(tmp_path.glob("outputs/gpu_telemetry/*/manifest.json")).read_text()
        )
        assert manifest["scientific_exit_code"] == 143
        assert manifest["completed_at"] is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_shared_gpu_wrapper_enables_telemetry_and_keeps_worker_identity(tmp_path):
    import os
    from pathlib import Path

    run = tmp_path / "run"
    source = tmp_path / "source"
    binaries = tmp_path / "bin"
    for directory in (run, source, binaries):
        directory.mkdir()
    sif = tmp_path / "image.sif"
    sif.touch()
    capture = tmp_path / "args"
    executable = binaries / "singularity"
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\nexit 9\n')
    executable.chmod(0o755)
    env = os.environ | {
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "TMPDIR": str(tmp_path),
        "CAPTURE": str(capture),
        "EPIFORECASTER_WORKER_ID": "6",
        "CUDA_VISIBLE_DEVICES": "3",
    }
    result = subprocess.run(
        [
            "bash",
            str(Path("scripts/workflow/mn5_container_exec.sh").resolve()),
            "--run-root",
            str(run),
            "--source-root",
            str(source),
            "--sif",
            str(sif),
            "--nv",
            "--",
            "command",
            "argument with spaces",
        ],
        env=env,
    )
    args = capture.read_text().splitlines()
    assert result.returncode == 9
    assert "scripts.workflow.gpu_telemetry" in args
    assert "EPIFORECASTER_WORKER_ID=6" in args
    assert "CUDA_VISIBLE_DEVICES=3" in args
    assert args[-2:] == ["command", "argument with spaces"]

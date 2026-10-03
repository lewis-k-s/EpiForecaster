"""Run a scientific command with optional, low-frequency GPU device telemetry."""

from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import uuid


@dataclass(frozen=True)
class TelemetryConfig:
    interval_seconds: float = 10.0
    query_timeout_seconds: float = 2.0


FIELDS = (
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
)
COLUMNS = (
    "timestamp_utc",
    "gpu_uuid",
    "gpu_utilization_percent",
    "memory_controller_utilization_percent",
    "memory_used_mib",
    "memory_total_mib",
    "power_watts",
    "temperature_celsius",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def visible_gpu_uuids() -> list[str]:
    """Resolve CUDA ordinals to UUIDs without creating a CUDA context.

    Never fall back to a node-wide nvidia-smi query. CUDA's driver enumeration
    applies CUDA_VISIBLE_DEVICES, unlike nvidia-smi's physical index namespace.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or visible == "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES is missing or disables CUDA")
    if "MIG-" in visible:
        raise RuntimeError("MIG telemetry is not supported by this full-GPU sampler")
    driver = ctypes.CDLL("libcuda.so.1")

    def checked(code: int) -> None:
        if code != 0:
            raise RuntimeError(f"CUDA device enumeration failed: {code}")

    checked(driver.cuInit(0))
    count = ctypes.c_int()
    checked(driver.cuDeviceGetCount(ctypes.byref(count)))
    devices = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        checked(driver.cuDeviceGet(ctypes.byref(device), ordinal))
        value = (ctypes.c_ubyte * 16)()
        uuid_query = getattr(driver, "cuDeviceGetUuid_v2", driver.cuDeviceGetUuid)
        checked(uuid_query(ctypes.byref(value), device))
        devices.append("GPU-" + str(uuid.UUID(bytes=bytes(value))))
    if not devices:
        raise RuntimeError("No CUDA devices are visible")
    return devices


def query_gpus(devices: list[str], timeout: float) -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + ",".join(devices),
            "--query-gpu=" + ",".join(FIELDS),
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )
    rows = []
    for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        if len(row) != len(FIELDS) or row[0] not in devices:
            raise ValueError("Unexpected GPU telemetry identity or column count")
        rows.append([now(), *[cell.strip() for cell in row]])
    if len(rows) != len(devices) or len({r[1] for r in rows}) != len(devices):
        raise ValueError("GPU telemetry returned missing or duplicate devices")
    return rows


class Sampler:
    """One sampler per worker invocation; failures do not stop the workload."""

    def __init__(self, run_root: Path, config: TelemetryConfig = TelemetryConfig()):
        self.config = config
        self.stop_event = threading.Event()
        attempt_file = run_root / ".state/active_execution"
        attempt = attempt_file.read_text().strip() if attempt_file.exists() else "local"
        # UUID suffix prevents retries and repeated worker commands colliding.
        self.root = run_root / "outputs/gpu_telemetry" / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.metadata = {
            "schema_version": 1,
            "execution_attempt_id": attempt,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "step_id": os.environ.get("SLURM_STEP_ID"),
            "rank": os.environ.get("SLURM_PROCID"),
            "worker_id": os.environ.get("EPIFORECASTER_WORKER_ID"),
            "started_at": now(),
            "completed_at": None,
            "status": "running",
            "interval_seconds": config.interval_seconds,
            "samples": 0,
            "errors": 0,
            "gpu_uuids": [],
            "artifact": "gpu_samples.csv",
        }
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.publish()

    def publish(self) -> None:
        temporary = self.root / "manifest.json.tmp"
        temporary.write_text(json.dumps(self.metadata, indent=2))
        temporary.replace(self.root / "manifest.json")

    def sample(self) -> None:
        try:
            devices = visible_gpu_uuids()
            self.metadata["gpu_uuids"] = devices
            with (self.root / "gpu_samples.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(COLUMNS)
                while not self.stop_event.is_set():
                    try:
                        rows = query_gpus(devices, self.config.query_timeout_seconds)
                        writer.writerows(rows)
                        handle.flush()
                        self.metadata["samples"] += len(rows)
                    except Exception as exc:
                        self.metadata["errors"] += 1
                        self.metadata["last_error"] = str(exc)
                        if self.metadata["errors"] == 1:
                            print(f"GPU telemetry warning: {exc}", file=sys.stderr)
                    self.publish()
                    self.stop_event.wait(self.config.interval_seconds)
        except Exception as exc:
            self.metadata["status"] = "unavailable"
            self.metadata["last_error"] = str(exc)
            print(f"GPU telemetry unavailable: {exc}", file=sys.stderr)

    def start(self) -> None:
        self.thread.start()

    def close(self, scientific_status: int) -> None:
        self.stop_event.set()
        self.thread.join()
        if self.metadata["status"] != "unavailable":
            self.metadata["status"] = (
                "complete" if not self.metadata["errors"] else "partial"
            )
        self.metadata["completed_at"] = now()
        self.metadata["scientific_exit_code"] = scientific_status
        artifact = self.root / "gpu_samples.csv"
        if artifact.exists():
            self.metadata["artifact_sha256"] = hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest()
        self.publish()


def run_command(command: list[str], run_root: Path) -> int:
    """Forward signals to the workload, stop sampling, preserve its exit status."""
    child = subprocess.Popen(command, start_new_session=True)
    sampler = None
    previous_handlers = {}

    def forward(signum, _frame):
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, forward)
    status = 1
    try:
        try:
            sampler = Sampler(run_root)
            sampler.start()
        except Exception as exc:
            print(f"GPU telemetry unavailable: {exc}", file=sys.stderr)
            sampler = None
        code = child.wait()
        status = code if code >= 0 else 128 - code
    finally:
        if sampler is not None:
            try:
                sampler.close(status)
            except Exception as exc:
                print(f"GPU telemetry close warning: {exc}", file=sys.stderr)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a scientific command is required")
    try:
        status = run_command(command, args.run_root)
    except OSError as exc:
        print(f"Could not launch scientific command: {exc}", file=sys.stderr)
        status = 127
    raise SystemExit(status)


if __name__ == "__main__":
    main()

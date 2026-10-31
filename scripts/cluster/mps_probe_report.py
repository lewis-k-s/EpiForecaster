"""Summarize an MPS HPO probe: per-job slowdown, aggregate speedup, GPU use.

Reads PROBE_ROOT/results/<mode>-k<K>/ written by mps_hpo_probe.sh. The baseline
is plain-k1 (one trial on the GPU, the production layout).

Per-job slowdown is the median steady epoch time of a condition divided by the
baseline epoch time. Aggregate speedup is K / slowdown: the trial-epochs
finished per unit time relative to the baseline. A condition meets the
no-regression goal when its slowdown is at most --max-slowdown.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from datetime import datetime, timedelta
from pathlib import Path

import click

_STEP_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) .*Epoch (?P<epoch>\d+) \| "
    r"Step (?P<step>\d+) \|.*SPS:\s*(?P<sps>[\d.]+)"
)
_BASELINE = "plain-k1"


def _parse_log_time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S,%f")


def _parse_smi_time(text: str) -> datetime:
    return datetime.strptime(text.strip(), "%Y/%m/%d %H:%M:%S.%f")


def _worker_series(log_path: Path) -> dict[str, object]:
    """Return epoch start times and per-step SPS values from one worker log."""
    starts: dict[int, datetime] = {}
    sps: dict[int, list[float]] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        match = _STEP_LINE.match(line)
        if match is None:
            continue
        epoch = int(match["epoch"])
        stamp = _parse_log_time(match["ts"])
        # Step counters are global, so the first logged step marks the epoch.
        starts.setdefault(epoch, stamp)
        sps.setdefault(epoch, []).append(float(match["sps"]))
    return {"starts": starts, "sps": sps}


def _epoch_times(series: dict[str, object], epochs: int) -> list[float]:
    """Epochs 1..N-2 have a following epoch start; epoch 0 includes compile."""
    starts: dict[int, datetime] = series["starts"]  # type: ignore[assignment]
    return [
        (starts[e + 1] - starts[e]).total_seconds()
        for e in range(1, epochs - 1)
        if e in starts and e + 1 in starts
    ]


def _timing_medians(condition_dir: Path) -> dict[str, float]:
    """Median data-load and model-step times after epoch 0, over all workers."""
    dataload: list[float] = []
    model: list[float] = []
    for path in condition_dir.glob("trials/**/training_dynamics.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("epoch", 0) < 1:
                continue
            if "time_dataload_s" in row:
                dataload.append(float(row["time_dataload_s"]))
            if "time_model_step_s" in row:
                model.append(float(row["time_model_step_s"]))
    return {
        "dataload_ms": 1e3 * statistics.median(dataload) if dataload else float("nan"),
        "model_step_ms": 1e3 * statistics.median(model) if model else float("nan"),
    }


def _gpu_window(
    condition_dir: Path, window: tuple[datetime, datetime]
) -> dict[str, float]:
    """Mean GPU use over the window when every worker is in steady state."""
    rows: list[tuple[datetime, float, float, float, float]] = []
    with (condition_dir / "gpu_samples.csv").open() as handle:
        for record in csv.reader(handle):
            if len(record) < 5:
                continue
            try:
                rows.append(
                    (
                        _parse_smi_time(record[0]),
                        float(record[1]),
                        float(record[2]),
                        float(record[3]),
                        float(record[4]),
                    )
                )
            except ValueError:
                continue
    if not rows:
        return {}
    # Container logs may use a different time zone from the host sampler.
    first_log, last_log = window
    offset = round((first_log - rows[0][0]).total_seconds() / 3600.0)
    shift = timedelta(hours=offset)
    inside = [r for r in rows if first_log <= r[0] + shift <= last_log]
    if not inside:
        inside = rows
    return {
        "sm_pct_mean": statistics.fmean(r[1] for r in inside),
        "sm_pct_zero_frac": sum(r[1] == 0 for r in inside) / len(inside),
        "mem_ctl_pct_mean": statistics.fmean(r[2] for r in inside),
        "mem_used_mib_max": max(r[3] for r in rows),
        "power_w_mean": statistics.fmean(r[4] for r in inside),
        "window_samples": float(len(inside)),
    }


def _summarize(condition_dir: Path) -> dict[str, object] | None:
    meta_path = condition_dir / "condition.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    epochs = int(meta["epochs"])
    workers = int(meta["workers"])
    series = [
        _worker_series(path) for path in sorted(condition_dir.glob("worker-*.log"))
    ]
    times = [_epoch_times(item, epochs) for item in series]
    flat = [t for worker_times in times for t in worker_times]
    sps = [
        v
        for item in series
        for epoch, values in item["sps"].items()  # type: ignore[union-attr]
        if epoch >= 1
        for v in values
    ]
    steady_starts = [
        item["starts"][e]  # type: ignore[index]
        for item in series
        for e in (1, epochs - 1)
        if e in item["starts"]  # type: ignore[operator]
    ]
    result: dict[str, object] = {
        "condition": condition_dir.name,
        "mode": meta["mode"],
        "workers": workers,
        "failed": bool(meta["failed"]),
        "epoch_time_s": statistics.median(flat) if flat else float("nan"),
        "epoch_time_samples": len(flat),
        "sps_median": statistics.median(sps) if sps else float("nan"),
        **_timing_medians(condition_dir),
    }
    if steady_starts and (condition_dir / "gpu_samples.csv").exists():
        low = max(
            item["starts"][1]
            for item in series
            if 1 in item["starts"]  # type: ignore[index,operator]
        )
        high = min(
            item["starts"][epochs - 1]  # type: ignore[index]
            for item in series
            if epochs - 1 in item["starts"]  # type: ignore[operator]
        )
        result.update(_gpu_window(condition_dir, (low, high)))
    return result


def _derive(rows: list[dict[str, object]], max_slowdown: float) -> None:
    baseline = next((r for r in rows if r["condition"] == _BASELINE), None)
    for row in rows:
        if baseline is None:
            continue
        slowdown = float(row["epoch_time_s"]) / float(baseline["epoch_time_s"])
        row["slowdown"] = slowdown
        row["aggregate_speedup"] = int(row["workers"]) / slowdown  # type: ignore[call-overload]
        row["meets_goal"] = (not row["failed"]) and slowdown <= max_slowdown


def _markdown(rows: list[dict[str, object]], max_slowdown: float) -> str:
    header = (
        "| condition | K | epoch s | slowdown | agg. speedup | ok | SM % | "
        "zero-SM | mem MiB | W | sps | load ms | step ms |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            "| {condition} | {workers} | {epoch_time_s:.1f} | {slowdown:.2f} | "
            "{aggregate_speedup:.2f} | {ok} | {sm_pct_mean:.0f} | "
            "{sm_pct_zero_frac:.0%} | {mem_used_mib_max:.0f} | {power_w_mean:.0f} | "
            "{sps_median:.0f} | {dataload_ms:.1f} | {model_step_ms:.1f} |".format(
                ok="yes" if r.get("meets_goal") else "NO",
                **{
                    k: r.get(k, float("nan"))
                    for k in (
                        "condition",
                        "workers",
                        "epoch_time_s",
                        "slowdown",
                        "aggregate_speedup",
                        "sm_pct_mean",
                        "sm_pct_zero_frac",
                        "mem_used_mib_max",
                        "power_w_mean",
                        "sps_median",
                        "dataload_ms",
                        "model_step_ms",
                    )
                },
            )
        )
    note = f"\nGoal: per-job slowdown <= {max_slowdown:.2f} versus {_BASELINE}.\n"
    return header + "\n".join(lines) + note


@click.command()
@click.argument("probe_root", type=click.Path(exists=True, path_type=Path))
@click.option("--max-slowdown", type=float, default=1.2, show_default=True)
def main(probe_root: Path, max_slowdown: float) -> None:
    """Print and save the probe summary for PROBE_ROOT."""
    rows = [
        summary
        for path in sorted((probe_root / "results").iterdir())
        if path.is_dir() and (summary := _summarize(path)) is not None
    ]
    _derive(rows, max_slowdown)
    (probe_root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    report = _markdown(rows, max_slowdown)
    (probe_root / "summary.md").write_text(report)
    click.echo(report)


if __name__ == "__main__":
    main()

"""Execution-level regressions for workflow retries and node-local staging."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import xarray as xr
import yaml

from scripts.workflow import attempts, run_support

REPO_ROOT = Path(__file__).resolve().parents[1]


def _snakemake(snakefile: Path, target: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            str(Path(sys.executable).with_name("snakemake")),
            "--snakefile",
            str(snakefile),
            "--directory",
            str(snakefile.parent),
            "--workflow-profile",
            str(REPO_ROOT / "workflow/profiles/mn5"),
            "--runtime-source-cache-path",
            str(snakefile.parent / "source-cache"),
            target,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("kind", ["hpo", "ablation"])
@pytest.mark.parametrize("failure_status", [1, 75])
@pytest.mark.integration
def test_study_rule_preserves_progress_and_resumes(tmp_path, kind, failure_status):
    """Execute the actual worker rule with a failing stand-in for srun."""
    rules = (REPO_ROOT / "workflow/rules/studies.smk").read_text()
    initializer = rules[
        rules.index("if WORKFLOW_KIND in") : rules.index('if WORKFLOW_KIND == "hpo"')
    ]
    if kind == "hpo":
        worker = rules[
            rules.index('if WORKFLOW_KIND == "hpo"') : rules.index(
                "    rule select_best_trial:"
            )
        ]
        target = "launch_optuna_workers"
    else:
        worker = rules[
            rules.index('elif WORKFLOW_KIND == "ablation"') : rules.index(
                "    rule finalize_ablation:"
            )
        ]
        worker = worker.replace("elif ", "if ", 1)
        target = "launch_ablation_workers"
    launcher = tmp_path / "worker.sh"
    launcher.write_text(f"""#!/bin/bash
set -eu
if [ ! -e first-attempt ]; then
  echo completed-trial >> study.journal
  echo complete > worker.done
  touch first-attempt
  exit {failure_status}
fi
test "$(cat study.journal)" = completed-trial
test -s worker.done
echo resumed >> study.journal
""")
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "dependencies.json").write_text("{}")
    prelude = f"""
from pathlib import Path
RUN_ROOT = Path({str(tmp_path)!r})
SOURCE_ROOT = RUN_ROOT
WORKFLOW_KIND = {kind!r}
JOURNAL = RUN_ROOT / "study.journal"
WORKERS_COMPLETE = RUN_ROOT / "complete.done"
DATA_MANIFEST = RUN_ROOT / "data.json"
DEPENDENCY_CONTRACTS = {{{kind!r}: RUN_ROOT / "dependencies.json"}}
STUDY_WORKERS = 1
CPUS_PER_TASK = 20
SIF = "unused"
config = {{}}
def hpo_worker_arguments(config): return []
def ablation_worker_arguments(config): return []
def launch_parallel_workers_command(*args, **kwargs): return {("bash " + shlex.quote(str(launcher)))!r}
"""
    snakefile = tmp_path / "Snakefile"
    snakefile.write_text(prelude + initializer + worker)
    first = _snakemake(snakefile, target)
    assert first.returncode != 0
    assert (tmp_path / "study.journal").read_text() == "completed-trial\n"
    assert (tmp_path / "worker.done").is_file()
    assert not (tmp_path / "complete.done").exists()
    second = _snakemake(snakefile, target)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (tmp_path / "study.journal").read_text() == "completed-trial\nresumed\n"
    third = _snakemake(snakefile, target)
    assert third.returncode == 0, third.stdout + third.stderr
    assert "Nothing to be done" in third.stdout + third.stderr


def test_completed_attempt_restores_missing_outputs_and_rejects_corruption(tmp_path):
    run = tmp_path / "run"
    root = run / ".state/attempts/stage"
    attempt = attempts.new_attempt(root)
    payload = attempt / "payload"
    payload.write_bytes(b"completed work")
    output = run / "outputs/result"
    identity = {"config": "abc"}
    attempts.complete_attempt(attempt, identity, {output: payload}, run)
    assert os.path.samefile(output, payload)
    output.unlink()
    assert attempts.restore_attempt(root, identity, run)
    assert output.read_bytes() == b"completed work"
    output.unlink()
    payload.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        attempts.restore_attempt(root, identity, run)
    assert not output.exists()


def test_crossval_retries_only_incomplete_attempt_and_reuses_completed_fold(
    tmp_path, monkeypatch
):
    run = tmp_path / "run"
    source = tmp_path / "source"
    run.mkdir()
    source.mkdir()
    (source / "split.json").write_text("{}")
    config = {
        "crossval": {
            "campaign_id": "test",
            "folds": 2,
            "model_seed": 42,
            "base_config": "train.yaml",
            "split_manifest": "split.json",
        },
        "evaluation": {"split": "test", "granular": False},
    }
    config_path = run / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    calls = []

    def train(path, source_root, attempt):
        calls.append(attempt)
        checkpoint, model = run_support.training_artifact_paths(
            run_support.load_yaml(path), attempt
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        if len(calls) == 1:
            raise RuntimeError("preempted")
        run_support.write_json(
            model / "training_manifest.json", {"training_dynamics": {}}
        )

    monkeypatch.setattr(run_support, "train_model", train)
    with pytest.raises(RuntimeError, match="preempted"):
        run_support.train_crossval_fold(config_path, source, run, 0)
    failed_files = {p: p.read_bytes() for p in calls[0].rglob("*") if p.is_file()}
    manifest = run_support.train_crossval_fold(config_path, source, run, 0)
    checkpoint = Path(json.loads(manifest.read_text())["checkpoint"])
    checkpoint.unlink()  # Simulate lost output and Snakemake cleanup of siblings.
    manifest.unlink()
    run_support.train_crossval_fold(config_path, source, run, 0)
    assert checkpoint.read_bytes() == b"checkpoint"
    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert all(path.read_bytes() == content for path, content in failed_files.items())


def test_preprocessing_retry_and_restore_do_not_repeat_completed_work(
    tmp_path, monkeypatch
):
    import data.preprocess

    run = tmp_path / "run"
    source = tmp_path / "source"
    run.mkdir()
    source.mkdir()
    (source / "preprocess.yaml").write_text("dataset_name: tiny\n")
    config_path = run / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "workflow": {"kind": "preprocess"},
                "preprocessing": {
                    "base_config": "preprocess.yaml",
                    "dataset_name": "tiny",
                },
            }
        )
    )
    monkeypatch.setattr(run_support, "validate_workflow_contract", lambda config: None)
    monkeypatch.setattr(
        data.preprocess.PreprocessingConfig,
        "from_file",
        lambda path: SimpleNamespace(**yaml.safe_load(path.read_text())),
    )
    calls = []

    class Pipeline:
        def __init__(self, config):
            self.config = config

        def run(self):
            path = Path(self.config.output_path) / "tiny-dated.nc"
            path.parent.mkdir(parents=True)
            calls.append(path)
            if len(calls) == 1:
                path.write_bytes(b"partial")
                raise RuntimeError("preempted")
            xr.Dataset({"cases": ("date", [1.0, 2.0])}).to_netcdf(path)
            return path

    monkeypatch.setattr(data.preprocess, "OfflinePreprocessingPipeline", Pipeline)
    monkeypatch.setattr(
        run_support, "_processed_dataset_qa", lambda *args: {"ok": True}
    )
    monkeypatch.setattr(run_support, "_handoff_provenance_qa", lambda *args: None)
    with pytest.raises(RuntimeError, match="preempted"):
        run_support.run_preprocessing(config_path, source, run)
    marker = run_support.run_preprocessing(config_path, source, run)
    output = run / "outputs/preprocessing/tiny.nc"
    original = output.read_bytes()
    for path in output.parent.iterdir():
        path.unlink()
    run_support.run_preprocessing(config_path, source, run)
    assert output.read_bytes() == original
    assert marker.is_file()
    assert len(calls) == 2
    assert calls[0].read_bytes() == b"partial"


def _cache_environment(tmp_path):
    run = tmp_path / "run"
    source = tmp_path / "source"
    fake_bin = tmp_path / "bin"
    for path in [
        run / ".state/dvc-workspace/data/files",
        source / "data/files",
        fake_bin,
    ]:
        path.mkdir(parents=True)
    (run / "config.yaml").write_text("source: immutable\n")
    (run / ".state/data_manifest.json").write_text('{"data": "identity-1"}')
    (run / ".state/dvc_bind_roots.txt").write_text("data/files\n")
    (run / ".state/dvc-workspace/data/files/data.nc").write_bytes(b"data")
    (source / "data/files/input.dvc").write_text("pointer")
    (tmp_path / "runtime.sif").touch()
    singularity = fake_bin / "singularity"
    singularity.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE-$SLURM_STEP_ID"\n'
    )
    singularity.chmod(0o755)
    real_rsync = shutil.which("rsync")
    rsync = fake_bin / "rsync"
    rsync.write_text(
        f'#!/bin/bash\nif [ "$1" != "--server" ]; then echo copy >> "$COPY_LOG"; fi\nif [ -f "$FAIL_COPY" ]; then exit 9; fi\nexec {shlex.quote(real_rsync)} "$@"\n'
    )
    rsync.chmod(0o755)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TMPDIR": str(tmp_path / "scratch"),
        "SLURM_JOB_ID": "12345",
        "SLURM_PROCID": "0",
        "EPIFORECASTER_NODE_CACHE_ROOT": str(tmp_path / "cache"),
        "COPY_LOG": str(tmp_path / "copies"),
        "CAPTURE": str(tmp_path / "binds"),
        "FAIL_COPY": str(tmp_path / "fail-copy"),
    }
    command = [
        "bash",
        str(REPO_ROOT / "scripts/workflow/mn5_container_exec.sh"),
        "--run-root",
        str(run),
        "--source-root",
        str(source),
        "--sif",
        str(tmp_path / "runtime.sif"),
        "--stage-data",
        "--",
        "true",
    ]
    return run, env, command


def _clean_cache(run, env):
    return subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/workflow/node_input_cache.sh"),
            "clean",
            str(run),
        ],
        env=env,
        capture_output=True,
        text=True,
    )


def test_concurrent_ranks_and_later_stage_share_one_copy(tmp_path):
    run, env, command = _cache_environment(tmp_path)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(
                    lambda step: subprocess.run(
                        command,
                        env=env | {"SLURM_STEP_ID": str(step)},
                        capture_output=True,
                        text=True,
                    ),
                    range(4),
                )
            )
        assert all(r.returncode == 0 for r in results), [
            (r.stdout, r.stderr) for r in results
        ]
        later = subprocess.run(
            command, env=env | {"SLURM_STEP_ID": "5"}, capture_output=True, text=True
        )
        assert later.returncode == 0, later.stderr
        assert (tmp_path / "copies").read_text().splitlines() == ["copy", "copy"]
        binds = [
            [
                line
                for line in (tmp_path / f"binds-{step}").read_text().splitlines()
                if ":/workspace/source/data/files:ro" in line
            ]
            for step in [0, 1, 2, 3, 5]
        ]
        assert all(value == binds[0] for value in binds)
        cached = Path(binds[0][0].split(":")[0]) / "data.nc"
        assert cached.read_bytes() == b"data"
        assert not cached.stat().st_mode & 0o222
        (run / ".state/data_manifest.json").write_text('{"data": "identity-2"}')
        changed = subprocess.run(
            command, env=env | {"SLURM_STEP_ID": "6"}, capture_output=True, text=True
        )
        assert changed.returncode == 0, changed.stderr
        assert len((tmp_path / "copies").read_text().splitlines()) == 4
    finally:
        assert _clean_cache(run, env).returncode == 0
    assert not (tmp_path / "cache/12345/run").exists()


def test_interrupted_copy_is_not_reused(tmp_path):
    run, env, command = _cache_environment(tmp_path)
    try:
        (tmp_path / "fail-copy").touch()
        failed = subprocess.run(
            command, env=env | {"SLURM_STEP_ID": "0"}, capture_output=True, text=True
        )
        assert failed.returncode == 9
        assert not list((tmp_path / "cache").rglob(".ready"))
        (tmp_path / "fail-copy").unlink()
        success = subprocess.run(
            command, env=env | {"SLURM_STEP_ID": "1"}, capture_output=True, text=True
        )
        assert success.returncode == 0, success.stderr
        assert len(list((tmp_path / "cache").rglob(".ready"))) == 1
    finally:
        assert _clean_cache(run, env).returncode == 0


def test_crossval_schedules_each_fold_without_a_training_barrier(tmp_path):
    rules = (REPO_ROOT / "workflow/rules/studies.smk").read_text()
    rules = rules[
        rules.index('elif WORKFLOW_KIND == "crossval"') : rules.index(
            "    rule finalize_crossval:"
        )
    ]
    rules = rules.replace("elif ", "if ", 1)
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "dependencies.json").write_text("{}")
    worker = tmp_path / "worker.py"
    worker.write_text("""
from pathlib import Path
import json, sys, time
root = Path(sys.argv[1])
stage, fold = sys.argv[2:]
fold_root = root / "fold-" / fold
with (root / "events").open("a") as handle:
    handle.write(f"{stage} {fold}\\n")
if stage == "training":
    if fold == "000":
        deadline = time.monotonic() + 15
        while not (root / "evaluation-001.done").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("Evaluation was blocked by another training fold")
            time.sleep(0.02)
    names = ["outputs/training/training/model/checkpoints/best_model.pt",
             "outputs/training/training/model/training_manifest.json",
             "config.yaml", "fold_training_manifest.json"]
else:
    assert (fold_root / "fold_training_manifest.json").exists()
    names = ["outputs/evaluation/test_summary.json",
             "outputs/evaluation/test_node_metrics.csv",
             "outputs/evaluation/evaluation_manifest.json", "fold_manifest.json"]
for name in names:
    path = fold_root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
if stage == "evaluation":
    (root / f"evaluation-{fold}.done").touch()
""")
    snakefile = tmp_path / "Snakefile"
    prelude = f"""
from pathlib import Path
import shlex
RUN_ROOT = Path({str(tmp_path)!r})
WORKFLOW_KIND = "crossval"
DATA_MANIFEST = RUN_ROOT / "data.json"
DEPENDENCY_CONTRACTS = {{stage: RUN_ROOT / "dependencies.json" for stage in ("crossval", "evaluation")}}
CROSSVAL_FOLD_ROOT = RUN_ROOT / "fold-" / "{{fold}}"
CROSSVAL_EVALUATION_ARTIFACTS = [str(CROSSVAL_FOLD_ROOT / "outputs/evaluation" / name) for name in ("test_summary.json", "test_node_metrics.csv")]
WORKER_IDS = ["000", "001"]
CPUS_PER_TASK = 20
def crossval_fold_command(stage, fold):
    return shlex.join([{sys.executable!r}, {str(worker)!r}, str(RUN_ROOT), stage, fold])
rule all:
    input:
        expand(str(CROSSVAL_FOLD_ROOT / "fold_manifest.json"), fold=WORKER_IDS),
        expand(CROSSVAL_EVALUATION_ARTIFACTS, fold=WORKER_IDS)
"""
    snakefile.write_text(prelude + rules)
    result = _snakemake(snakefile, "all")
    assert result.returncode == 0, result.stdout + result.stderr
    events = (tmp_path / "events").read_text().splitlines()
    assert sorted(events) == [
        "evaluation 000",
        "evaluation 001",
        "training 000",
        "training 001",
    ]
    (tmp_path / "fold-/001/outputs/evaluation/test_node_metrics.csv").unlink()
    result = _snakemake(snakefile, "all")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "events").read_text().splitlines() == events + ["evaluation 001"]


def test_preprocessing_dataset_is_a_required_dag_output(tmp_path):
    rules = (REPO_ROOT / "workflow/rules/preprocessing.smk").read_text()
    rules = rules[: rules.index("    rule publish_preprocessed:")]
    for name in ("data.json", "dependencies.json"):
        (tmp_path / name).write_text("{}")
    command = "touch dataset.nc config.yaml qa.json done"
    prelude = f"""
from pathlib import Path
RUN_ROOT = Path({str(tmp_path)!r})
SOURCE_ROOT = RUN_ROOT
WORKFLOW_KIND = "preprocess"
DATA_MANIFEST = RUN_ROOT / "data.json"
DEPENDENCY_CONTRACTS = {{"preprocessing": RUN_ROOT / "dependencies.json"}}
PREPROCESSING_DATASET = RUN_ROOT / "dataset.nc"
PREPROCESSING_EFFECTIVE_CONFIG = RUN_ROOT / "config.yaml"
PREPROCESSING_QA = RUN_ROOT / "qa.json"
PREPROCESSING_DONE = RUN_ROOT / "done"
CPUS_PER_TASK = 20
CONTAINER_CONFIG = CONTAINER_SOURCE = CONTAINER_RUN = RUN_ROOT
SIF = DVC_REMOTE = DVC_CACHE = "unused"
def container_command(*args, **kwargs): return {command!r}
def slurm_step(command, *args, **kwargs): return command
"""
    snakefile = tmp_path / "Snakefile"
    snakefile.write_text(prelude + rules)
    first = _snakemake(snakefile, "preprocess_dataset")
    assert first.returncode == 0, first.stdout + first.stderr
    (tmp_path / "dataset.nc").unlink()
    second = _snakemake(snakefile, "preprocess_dataset")
    assert second.returncode == 0, second.stdout + second.stderr
    assert (tmp_path / "dataset.nc").exists()


def test_cache_cleanup_waits_for_scientific_readers(tmp_path):
    run, env, command = _cache_environment(tmp_path)
    started = tmp_path / "started"
    release = tmp_path / "release"
    (tmp_path / "bin/singularity").write_text(
        f"#!/bin/bash\ntouch {shlex.quote(str(started))}\n"
        f"while [ ! -f {shlex.quote(str(release))} ]; do sleep 0.05; done\n"
    )
    process = subprocess.Popen(
        command,
        env=env | {"SLURM_STEP_ID": "0"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert started.exists()
        cleanup = _clean_cache(run, env)
        assert cleanup.returncode == 0
        assert "still in use" in cleanup.stderr
        assert list((tmp_path / "cache").rglob(".ready"))
    finally:
        release.touch()
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stdout + stderr
        assert _clean_cache(run, env).returncode == 0
    assert not (tmp_path / "cache/12345/run").exists()

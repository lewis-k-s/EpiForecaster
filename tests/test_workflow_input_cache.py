"""Allocation input-cache regressions ported from primary commit df2dfd7."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

REPO_ROOT = Path(__file__).resolve().parents[1]


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

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_batch_adapter_executes_explicit_shared_coordinator(tmp_path: Path) -> None:
    capture = tmp_path / "capture.txt"
    shared_coordinator = tmp_path / "shared_coordinator.sh"
    shared_coordinator.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" >"$COORDINATOR_CAPTURE"\n',
        encoding="utf-8",
    )
    shared_coordinator.chmod(0o755)

    subprocess.run(
        [
            str(REPO_ROOT / "scripts/workflow/snakemake_coordinator.sbatch"),
            str(shared_coordinator),
            "run-root",
            "snakemake-bin",
        ],
        env=os.environ | {"COORDINATOR_CAPTURE": str(capture)},
        check=True,
    )

    assert capture.read_text(encoding="utf-8") == "run-root\nsnakemake-bin\n"


def test_batch_adapter_rejects_relative_shared_coordinator() -> None:
    result = subprocess.run(
        [
            str(REPO_ROOT / "scripts/workflow/snakemake_coordinator.sbatch"),
            "relative/coordinator.sh",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "executable absolute path" in result.stderr

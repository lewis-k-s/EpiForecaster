"""Container image identity is checked against the bundle lock and template pin."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from scripts.workflow.run_config import (
    parse_build_metadata,
    validate_mn5_allocation,
    validate_workflow_contract,
    verify_container_identity,
)

LOCK = hashlib.sha256(b"lock").hexdigest()
SIF = "a" * 64
SIF_PATH = "/containers/epiforecaster-runtime-41550b2b5a2e47912950.sif"


def _metadata(lock: str = LOCK, sif: str = SIF) -> str:
    return f"dependency_key=k\nuv_lock_sha256={lock}\nsif_sha256={sif}\nhost=h\n"


def _verify(**overrides: object) -> None:
    arguments: dict[str, object] = {
        "resolved_sif": SIF_PATH,
        "metadata_text": _metadata(),
        "observed_sha256": SIF,
        "uv_lock_sha256": LOCK,
    }
    arguments.update(overrides)
    verify_container_identity(**arguments)  # type: ignore[arg-type]


def test_parse_build_metadata() -> None:
    assert parse_build_metadata("a=1\nbad\nb=x=y\n") == {"a": "1", "b": "x=y"}


def test_matching_image_passes_with_and_without_pin() -> None:
    _verify()
    _verify(expected_sha256=SIF)


def test_old_image_names_the_expected_lock() -> None:
    old_lock = hashlib.sha256(b"old").hexdigest()
    with pytest.raises(ValueError, match=f"uv_lock_sha256={LOCK}"):
        _verify(metadata_text=_metadata(lock=old_lock))


def test_missing_lock_record_fails() -> None:
    with pytest.raises(ValueError, match="unrecorded"):
        _verify(metadata_text="sif_sha256=" + SIF + "\n")


def test_metadata_for_another_image_fails() -> None:
    with pytest.raises(ValueError, match="does not match the observed image"):
        _verify(metadata_text=_metadata(sif="b" * 64))


def test_template_pin_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="container.expected_sha256"):
        _verify(expected_sha256="b" * 64)


def test_template_pin_must_be_hex_digest() -> None:
    document = yaml.safe_load(
        Path("workflow/config/mn5_ablation.yaml").read_text(encoding="utf-8")
    )
    document["container"] = {"expected_sha256": SIF}
    validate_workflow_contract(document)
    document["container"] = {"expected_sha256": "sha256:" + SIF}
    with pytest.raises(ValueError, match="expected_sha256"):
        validate_workflow_contract(document)


def test_site_default_is_the_digest_named_image() -> None:
    conf = Path(".cluster/mn5.conf").read_text(encoding="utf-8")
    assert "epiforecaster-runtime-41550b2b5a2e47912950.sif}" in conf
    assert "/epiforecaster-runtime.sif}" not in conf


def test_every_workflow_template_passes_the_contract_and_allocation_checks() -> None:
    templates = sorted(Path("workflow/config").glob("*.yaml"))
    assert templates
    checked = 0
    for path in templates:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        # Copy-and-fill templates are incomplete by design.
        if ".template." in path.name or (
            document["workflow"]["kind"] == "evaluate"
            and document["evaluation"].get("checkpoint") is None
        ):
            continue
        try:
            validate_workflow_contract(document)
            validate_mn5_allocation(document)
        except Exception as error:  # noqa: BLE001
            pytest.fail(f"{path.name}: {error}")
        checked += 1
    assert checked > 10

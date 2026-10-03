"""Test evidence contracts; report layout is outside core coverage."""

import json
from pathlib import Path
import sys

import pytest

from scripts.analysis import render_run_report as report


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "scores.csv").write_text(
        "source_id,wis\n0012,0.123456789\n<a>,0.5\n", encoding="utf-8"
    )
    # The renderer only links saved image bytes; image rendering needs no unit test.
    (tmp_path / "loss.png").write_bytes(b"saved figure")
    spec = {
        "schema_version": 1,
        "title": "Run <test>",
        "split_strategy": "region",
        "sections": [
            {
                "id": "scores",
                "title": "Saved scores",
                "items": [
                    {"kind": "table", "key": "scores", "title": "Head scores"},
                    {"kind": "figure", "key": "loss", "title": "Loss <curve>"},
                ],
            }
        ],
    }
    write_json(tmp_path / "report_spec.json", spec)
    write_json(
        tmp_path / "analysis.json", {"interpretation": "Do not publish this claim"}
    )
    path = tmp_path / "bundle_manifest.json"
    write_json(
        path,
        {
            "report_spec": "report_spec.json",
            "analysis": "analysis.json",
            "tables": {"scores": "scores.csv"},
            "figures": {"loss": "loss.png"},
        },
    )
    return path


def test_primary_is_deterministic_and_does_not_publish_analysis_prose(bundle):
    first, receipt = report.render_report(bundle)
    assert report.render_report(bundle) == (first, receipt)
    assert receipt["content"] == "tables_and_figures"
    assert "Automatic tables and figures" in first
    assert "Do not publish this claim" not in first
    assert "0012" in first
    assert "&lt;a&gt;" in first
    assert "0.123456789" in first
    assert "Run &lt;test&gt;" in first


def test_copied_source_hash_is_checked(bundle):
    root = bundle.parent
    (root / "source.json").write_text("{}", encoding="utf-8")
    (root / "inputs.json").write_text(
        json.dumps([{"copy": "source.json", "sha256": "wrong"}]), encoding="utf-8"
    )
    manifest = report.read_json(bundle)
    manifest["report_inputs"] = "inputs.json"
    write_json(bundle, manifest)
    with pytest.raises(ValueError, match="Input copy hash mismatch"):
        report.render_report(bundle)


def test_assets_cannot_escape_the_bundle(bundle):
    manifest = report.read_json(bundle)
    manifest["tables"]["scores"] = "../outside.csv"
    write_json(bundle, manifest)
    with pytest.raises(ValueError, match="inside the bundle"):
        report.render_report(bundle)


def test_temporal_layout_uses_explicit_split_strategy(bundle):
    path = bundle.parent / "report_spec.json"
    spec = report.read_json(path)
    spec["split_strategy"] = "temporal"
    write_json(path, spec)
    document, _ = report.render_report(bundle)
    assert "Temporal split" in document
    assert "Region / native-source split" not in document


def test_cli_refuses_to_overwrite_an_evidence_input(bundle, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run-report-render", "--bundle", str(bundle), "--output", str(bundle)],
    )
    original = bundle.read_bytes()
    with pytest.raises(SystemExit):
        report.main()
    assert bundle.read_bytes() == original

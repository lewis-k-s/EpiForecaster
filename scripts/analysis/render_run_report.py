"""Render saved report tables and figures.

This module reads an evidence bundle. It does not compute evaluation results,
select examples, run a model, or write qualitative findings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import quote


STYLE = """
:root { color-scheme: light; font-family: system-ui, sans-serif;
  color: #172033; background: #f4f7fb; line-height: 1.5; }
* { box-sizing: border-box; }
body { margin: 0; }
main { max-width: 1180px; margin: 32px auto; padding: 0 20px; }
header, section { background: white; border: 1px solid #dce2ec;
  border-radius: 3px; padding: 24px; margin-bottom: 20px; }
h1 { font-size: 2rem; margin: 0 0 12px; }
h2 { font-size: 1.35rem; margin: 0 0 16px; }
h3 { font-size: 1rem; margin: 20px 0 8px; }
a { color: #2457d6; }
nav { display: flex; flex-wrap: wrap; gap: 8px 20px; margin-top: 20px; }
.status { display: inline-block; background: #f4f7fb; padding: 5px 10px; }
.table-wrap { overflow-x: auto; margin: 12px 0; }
table { border-collapse: collapse; width: 100%; font-size: .875rem; }
th, td { padding: 8px 10px; text-align: left; vertical-align: top;
  border-bottom: 1px solid #dce2ec; overflow-wrap: anywhere; }
th { background: #f4f7fb; }
figure { margin: 20px 0; }
img { display: block; width: 100%; height: auto; }
figcaption, .source { font-size: .8rem; color: #5d687d; }
details { margin: 16px 0; }
summary { cursor: pointer; }
code { overflow-wrap: anywhere; }
@media print { main { max-width: none; margin: 0; }
  section { break-inside: avoid; } nav { display: none; } }
"""


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def bundle_path(root: Path, relative: str) -> Path:
    """Keep report assets inside the portable bundle, including symlink targets."""
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Asset must be inside the bundle: {relative}")
    if not path.is_file():
        raise ValueError(f"Missing report asset: {relative}")
    return path


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def evidence_identity(bundle_file: Path, bundle: dict, spec: dict) -> tuple[str, dict]:
    """Record the spec, declared evidence, figures, and table identities."""
    root = bundle_file.parent
    paths = {bundle_file.name, bundle["report_spec"]}
    for key in (
        "analysis",
        "provenance",
        "native_analysis",
        "native_provenance",
        "report_inputs",
    ):
        if key in bundle:
            paths.add(bundle[key])
    for group in ("tables", "figures"):
        paths.update(bundle.get(group, {}).values())
    paths.update(link["path"] for link in spec.get("links", []))
    # Copies can be checked without access to the original run or data stores.
    if "report_inputs" in bundle:
        inputs = json.loads(
            bundle_path(root, bundle["report_inputs"]).read_text(encoding="utf-8")
        )
        for item in inputs:
            if "copy" in item:
                path = bundle_path(root, item["copy"])
                if sha256(path) != item["sha256"]:
                    raise ValueError(f"Input copy hash mismatch: {item['copy']}")
                paths.add(item["copy"])
    hashes = {path: sha256(bundle_path(root, path)) for path in sorted(paths)}
    digest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return digest, hashes


def link(relative: str, label: str) -> str:
    return f'<a href="{html.escape(quote(relative, safe="/"), quote=True)}">{html.escape(label)}</a>'


def render_table(root: Path, relative: str, item: dict) -> str:
    with bundle_path(root, relative).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        columns = item.get("columns", fields)
        if not columns or any(column not in fields for column in columns):
            raise ValueError(f"Missing table columns: {relative}")
        labels = item.get("labels", {})
        headings = "".join(
            f'<th scope="col">{html.escape(labels.get(c, c.replace("_", " ")))}</th>'
            for c in columns
        )
        rows = []
        for row in reader:
            cells = []
            for column in columns:
                value = row[column] or "—"
                if column in item.get("decimals", {}) and row[column]:
                    value = f"{float(row[column]):.{item['decimals'][column]}f}"
                cells.append(f"<td>{html.escape(value)}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + headings
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + '<div class="source">'
        + link(relative, "Source CSV")
        + "</div>"
    )


def render_item(root: Path, bundle: dict, item: dict) -> str:
    title = html.escape(item["title"])
    if item["kind"] == "table":
        body = render_table(root, bundle["tables"][item["key"]], item)
    elif item["kind"] == "figure":
        relative = bundle["figures"][item["key"]]
        bundle_path(root, relative)
        url = html.escape(quote(relative, safe="/"), quote=True)
        caption = html.escape(item.get("caption", item["title"]))
        body = (
            f'<figure><a href="{url}"><img src="{url}" alt="{title}" '
            f'loading="lazy"></a><figcaption>{caption}</figcaption></figure>'
        )
    else:
        raise ValueError(f"Unknown report item kind: {item['kind']}")
    if item.get("collapsed", False):
        return f"<details><summary>{title}</summary>{body}</details>"
    return f"<h3>{title}</h3>{body}"


def render_report(bundle_file: Path) -> tuple[str, dict]:
    bundle = read_json(bundle_file)
    spec = read_json(bundle_path(bundle_file.parent, bundle["report_spec"]))
    if spec.get("schema_version") != 1:
        raise ValueError("Report spec must have schema_version=1")
    if spec.get("split_strategy") not in {"region", "temporal"}:
        raise ValueError("Report spec must declare region or temporal split_strategy")
    identities = [section["id"] for section in spec["sections"]]
    if len(set(identities)) != len(identities):
        raise ValueError("Report section IDs must be unique")
    if any(not re.fullmatch(r"[a-z][a-z0-9-]*", key) for key in identities):
        raise ValueError("Report section IDs must be simple HTML identifiers")
    digest, hashes = evidence_identity(bundle_file, bundle, spec)
    title = html.escape(spec["title"])
    split = {"region": "Region / native-source split", "temporal": "Temporal split"}[
        spec["split_strategy"]
    ]
    nav = "".join(
        f'<a href="#{section["id"]}">{html.escape(section["title"])}</a>'
        for section in spec["sections"]
    )
    blocks = []
    for section in spec["sections"]:
        content = "".join(
            render_item(bundle_file.parent, bundle, item) for item in section["items"]
        )
        blocks.append(
            f'<section id="{section["id"]}"><h2>{html.escape(section["title"])}</h2>'
            + content
            + "</section>"
        )
    links = " · ".join(
        link(entry["path"], entry["label"]) for entry in spec.get("links", [])
    )
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title><style>{STYLE}</style></head><body>"
        f'<main data-evidence-sha256="{digest}"><header><h1>{title}</h1>'
        f'<div>{split} · <span class="status">Automatic tables and figures</span></div>'
        f"<nav>{nav}</nav></header>"
        + "".join(blocks)
        + f'<footer><div class="source">{links}</div>'
        + f'<div class="source">Evidence SHA-256: <code>{digest}</code></div></footer>'
        + "</main></body></html>\n"
    )
    receipt = {
        "schema_version": 1,
        "evidence_sha256": digest,
        "input_sha256": hashes,
        "generator_sha256": sha256(Path(__file__)),
        "report_sha256": hashlib.sha256(document.encode()).hexdigest(),
        "content": "tables_and_figures",
    }
    return document, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="HTML output inside the bundle directory"
    )
    args = parser.parse_args()
    bundle_file = args.bundle.resolve()
    output = args.output or bundle_file.parent / "report.html"
    if output.resolve().parent != bundle_file.parent:
        parser.error(
            "HTML output must be beside the bundle manifest for relative asset links"
        )
    document, receipt = render_report(bundle_file)
    # Refuse to replace a source file with generated HTML or a receipt.
    protected = {
        bundle_path(bundle_file.parent, path) for path in receipt["input_sha256"]
    }
    destinations = [output.resolve(), output.with_suffix(".render.json").resolve()]
    if len(set(destinations)) != len(destinations) or any(
        p in protected for p in destinations
    ):
        parser.error(
            "Output paths must be distinct from each other and from report inputs"
        )
    output.write_text(document, encoding="utf-8")
    output.with_suffix(".render.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

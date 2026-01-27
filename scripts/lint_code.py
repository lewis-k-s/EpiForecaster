"""Generic lint skill using Ruff for code quality and formatting checks.

This script runs Ruff linting and optionally formatting checks, then outputs
results in a standardized skill envelope format.

Usage:
    lint_code.py <paths...> [options]
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from utils.skill_output import SkillOutputBuilder, print_output


@dataclass
class Violation:
    """A single linting violation."""

    code: str
    severity: str
    message: str
    filename: str
    location: dict[str, int]
    end_location: dict[str, int] | None = None
    fixable: bool = False
    url: str | None = None


@dataclass
class Fix:
    """Information about a fix."""

    code: str
    filename: str
    message: str
    applicability: str
    edits: list[dict[str, Any]]


def map_severity(code: str) -> str:
    """Map Ruff error code to severity level."""
    prefix = code.split(" ")[0] if " " in code else code[0]
    severity_map = {
        "E": "error",
        "W": "warning",
        "F": "lint",
        "I": "info",
        "N": "refactor",
        "UP": "upgrade",
        "B": "bugbear",
        "C": "complexity",
        "S": "security",
        "A": "flake8-builtins",
        "R": "refactor",
    }
    return severity_map.get(prefix, "unknown")


def run_ruff(
    paths: list[str],
    fix: bool = False,
    fix_only: bool = False,
    unsafe_fixes: bool = False,
    select: str | None = None,
    ignore: str | None = None,
    exclude: str | None = None,
    max_line_length: int | None = None,
) -> list[Violation]:
    """Run Ruff and return parsed violations."""
    cmd = ["ruff", "check", "--output-format", "json"]

    if fix:
        cmd.append("--fix")
    if fix_only:
        cmd.append("--fix-only")
    if unsafe_fixes:
        cmd.append("--unsafe-fixes")
    if select:
        cmd.extend(["--select", select])
    if ignore:
        cmd.extend(["--ignore", ignore])
    if exclude:
        cmd.extend(["--exclude", exclude])
    if max_line_length:
        cmd.extend(["--line-length", str(max_line_length)])

    cmd.extend(paths)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise FileNotFoundError("ruff not found. Install with: pip install ruff")

    if not result.stdout:
        return []

    violations = []
    for item in json.loads(result.stdout):
        violation = Violation(
            code=item.get("code", "UNKNOWN"),
            severity=map_severity(item.get("code", "UNKNOWN")),
            message=item.get("message", ""),
            filename=item.get("filename", ""),
            location={
                "row": item.get("location", {}).get("row", 0),
                "column": item.get("location", {}).get("column", 0),
            },
            end_location=(
                {
                    "row": item.get("end_location", {}).get("row", 0),
                    "column": item.get("end_location", {}).get("column", 0),
                }
                if item.get("end_location")
                else None
            ),
            fixable=item.get("fix") is not None,
            url=item.get("url"),
        )
        violations.append(violation)

    return violations


def run_format_check(
    paths: list[str], exclude: str | None = None
) -> list[dict[str, Any]]:
    """Check formatting with ruff format --check."""
    cmd = ["ruff", "format", "--check", "--diff"]
    if exclude:
        cmd.extend(["--exclude", exclude])
    cmd.extend(paths)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise FileNotFoundError("ruff not found. Install with: pip install ruff")

    if result.returncode == 0:
        return []

    files_with_issues = []
    lines = result.stdout.split("\n")
    for line in lines:
        if line.strip().startswith("Would reformat"):
            filename = line.replace("Would reformat", "").strip()
            if filename:
                files_with_issues.append({"filename": filename, "type": "formatting"})

    return files_with_issues


def format_violation(v: Violation) -> dict[str, Any]:
    """Format violation for JSON output."""
    data = {
        "code": v.code,
        "severity": v.severity,
        "message": v.message,
        "filename": v.filename,
        "location": v.location,
        "fixable": v.fixable,
    }

    if v.end_location:
        data["end_location"] = v.end_location
    if v.url:
        data["url"] = v.url

    return data


def print_violations(
    violations: list[Violation], formatting_issues: list[dict[str, Any]]
) -> None:
    """Print human-readable violations."""
    if not violations and not formatting_issues:
        print("No violations found!")
        return

    print("=" * 80)
    print("LINT VIOLATIONS")
    print("=" * 80)

    for v in violations:
        severity_emoji = {"error": "❌", "warning": "⚠️", "lint": "🔧", "info": "ℹ️"}.get(
            v.severity, "•"
        )
        print(
            f"\n{severity_emoji} {v.filename}:{v.location['row']}:{v.location['column']}"
        )
        print(f"  {v.code}: {v.message}")
        if v.url:
            print(f"  📚 {v.url}")
        if v.fixable:
            print("  ✨ Fixable")

    if formatting_issues:
        print("\n" + "=" * 80)
        print("FORMATTING ISSUES")
        print("=" * 80)
        for issue in formatting_issues:
            print(f"  • {issue['filename']} - needs reformatting")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run Ruff linting and formatting checks with skill envelope output"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to lint",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply auto-fixes",
    )
    parser.add_argument(
        "--fix-only",
        action="store_true",
        help="Only show fixable issues",
    )
    parser.add_argument(
        "--unsafe-fixes",
        action="store_true",
        help="Include unsafe fixes",
    )
    parser.add_argument(
        "--format-check",
        action="store_true",
        help="Also check formatting with ruff format --check",
    )
    parser.add_argument(
        "--exclude",
        help="Exclude files matching pattern",
    )
    parser.add_argument(
        "--select",
        help="Select specific rules (e.g., E,F,W)",
    )
    parser.add_argument(
        "--ignore",
        help="Ignore specific rules",
    )
    parser.add_argument(
        "--max-line-length",
        type=int,
        help="Override max line length",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Output as human-readable text (default: JSON)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Output compact JSON (no indentation)",
    )

    args = parser.parse_args()

    builder = SkillOutputBuilder(
        skill_name="lint",
        input_path=", ".join(args.paths),
    )

    try:
        # Run linting
        violations = run_ruff(
            paths=args.paths,
            fix=args.fix,
            fix_only=args.fix_only,
            unsafe_fixes=args.unsafe_fixes,
            select=args.select,
            ignore=args.ignore,
            exclude=args.exclude,
            max_line_length=args.max_line_length,
        )

        # Filter violations if fix-only mode
        if args.fix_only:
            violations = [v for v in violations if v.fixable]

        # Run format check if requested
        formatting_issues = []
        if args.format_check:
            formatting_issues = run_format_check(paths=args.paths, exclude=args.exclude)

        # Collect all affected files
        all_files = {v.filename for v in violations}
        all_files.update(issue["filename"] for issue in formatting_issues)

        # Build summary
        severity_counts: dict[str, int] = {}
        code_counts: dict[str, int] = {}
        for v in violations:
            severity_counts[v.severity] = severity_counts.get(v.severity, 0) + 1
            code_counts[v.code] = code_counts.get(v.code, 0) + 1

        fixable_count = sum(1 for v in violations if v.fixable)
        non_fixable_count = len(violations) - fixable_count

        data = {
            "files_checked": len(all_files),
            "total_violations": len(violations),
            "violations_by_severity": severity_counts,
            "violations_by_code": code_counts,
            "fixable_violations": fixable_count,
            "non_fixable_violations": non_fixable_count,
            "formatting_issues": len(formatting_issues),
            "files_with_violations": sorted(all_files),
            "violations": [format_violation(v) for v in violations],
            "formatting_issues_detail": formatting_issues,
        }

        # Add warning if fix was applied
        if args.fix and fixable_count > 0:
            builder.add_warning(f"Applied {fixable_count} auto-fixes")

        output = builder.success(data)

        if args.text:
            print_violations(violations, formatting_issues)
            print(
                f"Summary: {len(violations)} violations, {len(formatting_issues)} formatting issues"
            )
        else:
            indent = 0 if args.compact else 2
            print_output(output, indent=indent)

    except FileNotFoundError as e:
        print_output(builder.error("FileNotFoundError", str(e)))
    except json.JSONDecodeError as e:
        print_output(
            builder.error("JSONDecodeError", f"Failed to parse Ruff output: {e}")
        )
    except Exception as e:
        print_output(builder.error(type(e).__name__, str(e), {"traceback": str(e)}))


if __name__ == "__main__":
    main()

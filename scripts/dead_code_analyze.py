#!/usr/bin/env python3
"""
Dead code analysis tool using vulture static analysis enhanced with Git history.

This script analyzes Python code to identify unused functions, classes, imports,
and unreachable code. It enhances vulture's analysis with Git history to provide
context and risk assessment for cleanup decisions.
"""

import argparse
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from utils.skill_output import SkillOutputBuilder, print_output

try:
    import vulture
except ImportError:
    print("Error: vulture package not found. Install with: pip install vulture")
    sys.exit(1)


@dataclass
class DeadCodeItem:
    """Represents a single dead code finding."""

    filename: str
    line: int
    name: str
    typ: str
    confidence: int
    size: int
    message: str
    last_modified: str
    age_days: int
    commit_count: int
    risk_level: str
    recommendation: str


@dataclass
class AnalysisSummary:
    """Summary of dead code analysis."""

    total_files: int
    total_lines: int
    total_findings: int
    filtered_findings: int
    findings_by_confidence: dict
    findings_by_risk: dict
    recommendations: dict


def get_git_history(filename: str, line: Optional[int] = None) -> tuple[str, int, int]:
    """
    Get Git history for a file (and optionally a specific line).

    Returns:
        tuple: (last_modified_date, age_days, commit_count)
    """
    try:
        cmd = ["git", "log", "--format=%ct", "-1"]
        if line is not None:
            cmd.extend(["-L", f"{line},{line}:{filename}"])
        else:
            cmd.append(filename)

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(filename).parent, timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            timestamp = int(result.stdout.strip())
            last_modified = datetime.fromtimestamp(timestamp)
            age_days = (datetime.now() - last_modified).days

            commit_count_cmd = ["git", "log", "--oneline", "--count", filename]
            if line is not None:
                commit_count_cmd.extend(["-L", f"{line},{line}:{filename}"])

            commit_count_result = subprocess.run(
                commit_count_cmd,
                capture_output=True,
                text=True,
                cwd=Path(filename).parent,
                timeout=10,
            )

            commit_count = 0
            if commit_count_result.returncode == 0:
                try:
                    commit_count = int(commit_count_result.stdout.strip())
                except ValueError:
                    commit_count = 0

            return last_modified.strftime("%Y-%m-%d"), age_days, commit_count
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass

    return "unknown", 0, 0


def assess_risk(
    item: "vulture.Item", age_days: int, commit_count: int
) -> tuple[str, str]:
    """
    Assess risk level and provide recommendation for a dead code item.

    Returns:
        tuple: (risk_level, recommendation)
    """
    confidence = item.confidence

    if confidence == 100:
        if age_days >= 60:
            return "LOW", "SAFE_TO_REMOVE"
        else:
            return "MEDIUM", "INVESTIGATE"
    elif confidence >= 90:
        if age_days >= 60:
            return "LOW", "SAFE_TO_REMOVE"
        else:
            return "MEDIUM", "INVESTIGATE"
    elif confidence >= 80:
        if age_days >= 90:
            return "MEDIUM", "CONSIDER_DEPRECATE"
        else:
            return "MEDIUM", "INVESTIGATE"
    else:
        if age_days >= 180:
            return "MEDIUM", "CONSIDER_DEPRECATE"
        else:
            return "HIGH", "INVESTIGATE_FURTHER"


def count_lines_in_code(code_str: str) -> int:
    """Count non-empty, non-comment lines in code."""
    lines = code_str.split("\n")
    count = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def analyze_code(
    paths: List[str],
    min_confidence: int = 80,
    sort_by_size: bool = False,
    exclude: Optional[List[str]] = None,
    ignore_names: Optional[List[str]] = None,
    ignore_decorators: Optional[List[str]] = None,
) -> tuple[List[DeadCodeItem], AnalysisSummary]:
    """
    Run vulture analysis and enhance with Git history.

    Returns:
        tuple: (list of DeadCodeItem, AnalysisSummary)
    """
    v = vulture.Vulture(verbose=False)

    if exclude:
        v.exclude = exclude
    if ignore_names:
        v.ignore_names = ignore_names
    if ignore_decorators:
        v.ignore_decorators = ignore_decorators

    expanded_paths = []
    for path in paths:
        p = Path(path)
        if p.is_file() and p.suffix == ".py":
            expanded_paths.append(str(p))
        elif p.is_dir():
            expanded_paths.extend([str(f) for f in p.rglob("*.py")])

    if not expanded_paths:
        print(f"No Python files found in: {', '.join(paths)}")
        return [], AnalysisSummary(0, 0, 0, 0, {}, {}, {})

    v.scavenge(expanded_paths)
    items = v.get_unused_code(min_confidence=min_confidence)

    total_lines = 0
    for path in expanded_paths:
        try:
            total_lines += len(Path(path).read_text().split("\n"))
        except (OSError, UnicodeDecodeError):
            pass

    dead_code_items = []
    findings_by_confidence = defaultdict(int)
    findings_by_risk = defaultdict(int)
    recommendations = defaultdict(int)

    for item in items:
        if item.confidence < min_confidence:
            continue

        last_modified, age_days, commit_count = get_git_history(
            item.filename, item.first_lineno
        )

        risk_level, recommendation = assess_risk(item, age_days, commit_count)

        dead_code_item = DeadCodeItem(
            filename=item.filename,
            line=item.first_lineno,
            name=item.name,
            typ=item.typ,
            confidence=item.confidence,
            size=item.size,
            message=item.message,
            last_modified=last_modified,
            age_days=age_days,
            commit_count=commit_count,
            risk_level=risk_level,
            recommendation=recommendation,
        )

        dead_code_items.append(dead_code_item)

        conf_bucket = f"{item.confidence}%"
        findings_by_confidence[conf_bucket] += 1
        findings_by_risk[risk_level] += 1
        recommendations[recommendation] += 1

    if sort_by_size:
        dead_code_items.sort(key=lambda x: x.size, reverse=True)

    summary = AnalysisSummary(
        total_files=len(expanded_paths),
        total_lines=total_lines,
        total_findings=len(items),
        filtered_findings=len(dead_code_items),
        findings_by_confidence=dict(findings_by_confidence),
        findings_by_risk=dict(findings_by_risk),
        recommendations=dict(recommendations),
    )

    return dead_code_items, summary


def print_report(items: List[DeadCodeItem], summary: AnalysisSummary):
    """Print human-readable analysis report."""
    print("=" * 80)
    print("DEAD CODE ANALYSIS REPORT")
    print("=" * 80)
    print(
        f"Analyzed: {summary.total_files} files, {summary.total_lines:,} lines of code"
    )
    print(f"Vulture findings: {summary.total_findings} items")
    print(f"Filtered to min-confidence: {summary.filtered_findings} items")
    print()

    if not items:
        print("No dead code found!")
        return

    items_by_confidence = defaultdict(list)
    for item in items:
        if item.confidence >= 90:
            items_by_confidence["HIGH (90-100%)"].append(item)
        elif item.confidence >= 70:
            items_by_confidence["MEDIUM (70-89%)"].append(item)
        else:
            items_by_confidence["LOW (60-69%)"].append(item)

    for conf_group, group_items in items_by_confidence.items():
        if not group_items:
            continue

        print("=" * 80)
        print(f"{conf_group} - {len(group_items)} items")
        print("-" * 80)
        print(
            f"{'File:Line':<30} | {'Type':<12} | {'Name':<25} | {'Size':<6} | "
            f"{'Last Modified':<12} | {'Risk':<8}"
        )
        print("-" * 80)

        for item in group_items:
            filename_short = str(item.filename).replace(str(Path.cwd()), ".")
            print(
                f"{filename_short}:{item.line:<26} | {item.typ:<12} | "
                f"{item.name:<25} | {item.size:<6} | "
                f"{item.last_modified:<12} | {item.risk_level:<8}"
            )
        print("-" * 80)

    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("-" * 80)
    rec_descriptions = {
        "SAFE_TO_REMOVE": "Safe to remove (high confidence, old code)",
        "INVESTIGATE": "Investigate further (may be used dynamically)",
        "CONSIDER_DEPRECATE": "Consider deprecation (add TODO/FIXME)",
        "INVESTIGATE_FURTHER": "High priority investigation (low confidence, recent)",
    }
    for rec, count in summary.recommendations.items():
        desc = rec_descriptions.get(rec, rec)
        print(f"- {count:2d} items: {desc}")

    print()
    print("=" * 80)
    print("RISK SUMMARY")
    print("-" * 80)
    total = sum(summary.findings_by_risk.values())
    for risk, count in summary.findings_by_risk.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"- {risk:8s}: {count:2d} items ({pct:5.1f}%)")
    print()


def make_whitelist(paths: List[str]) -> None:
    """Generate a vulture whitelist by simulating usage of all detected items."""
    v = vulture.Vulture(verbose=False)

    expanded_paths = []
    for path in paths:
        p = Path(path)
        if p.is_file() and p.suffix == ".py":
            expanded_paths.append(str(p))
        elif p.is_dir():
            expanded_paths.extend([str(f) for f in p.rglob("*.py")])

    v.scavenge(expanded_paths)
    items = v.get_unused_code()

    print("# Vulture whitelist - generated to suppress false positives")
    print("# Add this file to vulture paths to whitelist these items")
    print()

    for item in items:
        if item.typ in ("attribute", "function", "method", "property"):
            module = Path(item.filename).stem
            print(f"from {module} import {item.name}")
            print(f"{item.name}")
        elif item.typ == "class":
            module = Path(item.filename).stem
            print(f"from {module} import {item.name}")
            print(f"{item.name}")


def main():
    """CLI entry point for dead-code-analyze."""
    parser = argparse.ArgumentParser(
        description="Find and analyze dead Python code using vulture"
    )
    parser.add_argument(
        "paths", nargs="+", help="Python files or directories to analyze"
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=80,
        help="Minimum confidence level (60-100, default: 80)",
    )
    parser.add_argument(
        "--sort-by-size",
        action="store_true",
        help="Sort results by code size (largest first)",
    )
    parser.add_argument(
        "--exclude", nargs="*", help="Exclude files matching these patterns"
    )
    parser.add_argument(
        "--ignore-names", nargs="*", help="Ignore names matching these patterns"
    )
    parser.add_argument(
        "--ignore-decorators", nargs="*", help="Ignore functions with these decorators"
    )
    parser.add_argument(
        "--make-whitelist",
        action="store_true",
        help="Generate a whitelist for all detected items",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Output results as human-readable text (default: JSON)",
    )
    parser.add_argument(
        "--compact", action="store_true", help="Output compact JSON (no indentation)"
    )

    args = parser.parse_args()

    if args.make_whitelist:
        make_whitelist(args.paths)
        return

    builder = SkillOutputBuilder(
        skill_name="dead-code-analyze",
        input_path=", ".join(args.paths),
    )

    try:
        items, summary = analyze_code(
            paths=args.paths,
            min_confidence=args.min_confidence,
            sort_by_size=args.sort_by_size,
            exclude=args.exclude,
            ignore_names=args.ignore_names,
            ignore_decorators=args.ignore_decorators,
        )

        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        data = make_serializable(asdict(summary))
        data["findings"] = [make_serializable(asdict(item)) for item in items]

        output = builder.success(data)

        if args.text:
            print_report(items, summary)
        else:
            indent = 0 if args.compact else 2
            print_output(output, indent=indent)

    except FileNotFoundError as e:
        print_output(builder.error("FileNotFoundError", str(e)))
    except Exception as e:
        print_output(builder.error(type(e).__name__, str(e), {"traceback": str(e)}))


if __name__ == "__main__":
    main()

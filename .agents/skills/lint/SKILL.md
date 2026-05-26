---
name: lint
description: Run Ruff linting and formatting checks on Python code, with machine-readable JSON output and optional auto-fix support.
allowed-tools: Bash, Read
---

# Lint

Generic linting skill using Ruff for code quality, style, and formatting checks.

## Quick Start

```
lint scripts/ models/
lint scripts/ --fix --text
lint scripts/ --format-check
lint . --select E,F,W --fix-only
```

## Purpose

Use this skill when you want to:
- **Check code quality**: Run linting on files or directories
- **Fix auto-fixable issues**: Apply Ruff's automatic fixes
- **Check formatting**: Verify code formatting with `ruff format`
- **Filter by severity**: Select specific rule categories
- **Generate structured reports**: Get machine-readable JSON for CI/CD integration

## Input Formats

### Paths
```
lint <file_or_dir> [<file_or_dir> ...]
```
Examples:
- `lint scripts/` - Lint entire scripts directory
- `lint models/epiforecaster.py` - Lint single file
- `lint .` - Lint entire repository

### Rule Selection

```bash
lint scripts/ --select E,F,W
lint . --ignore E501,W293
```

**Common rule codes:**
- `E` - Error (pyflakes)
- `F` - Pyflakes (unused imports, undefined names)
- `W` - Warning (pycodestyle)
- `I` - isort (import sorting)
- `N` - PEP8 naming conventions
- `UP` - pyupgrade (Python 2 to 3 upgrade)
- `B` - flake8-bugbear
- `C` - mccabe (complexity)
- `S` - flake8-bandit (security)

## Output Format

### JSON (default)
```json
{
  "ok": true,
  "name": "lint",
  "version": "1.0",
  "data": {
    "files_checked": 5,
    "total_violations": 12,
    "violations_by_severity": {
      "lint": 5,
      "warning": 4,
      "error": 2,
      "info": 1
    },
    "violations_by_code": {
      "F401": 2,
      "E501": 3,
      "W291": 2,
      "I001": 5
    },
    "fixable_violations": 8,
    "non_fixable_violations": 4,
    "formatting_issues": 0,
    "files_with_violations": [
      "scripts/analysis/analyze_gradnorm.py",
      "scripts/analysis/dead_code_analyze.py"
    ],
    "violations": [
      {
        "code": "F401",
        "severity": "lint",
        "message": "`json` imported but unused",
        "filename": "scripts/analysis/analyze_gradnorm.py",
        "location": {"row": 19, "column": 8},
        "fixable": true,
        "url": "https://docs.astral.sh/ruff/rules/unused-import"
      }
    ],
    "formatting_issues_detail": []
  },
  "warnings": [],
  "meta": {
    "latency_ms": 234.5,
    "timestamp": "2025-01-25T15:30:00Z",
    "input_path": "scripts/"
  }
}
```

### Text output (--text)
```
================================================================================
LINT VIOLATIONS
================================================================================

❌ scripts/analysis/analyze_gradnorm.py:19:8
  F401: `json` imported but unused
  📚 https://docs.astral.sh/ruff/rules/unused-import
  ✨ Fixable

⚠️ scripts/analysis/dead_code_analyze.py:11:8
  F401: `json` imported but unused
  📚 https://docs.astral.sh/ruff/rules/unused-import
  ✨ Fixable

Summary: 2 violations, 0 formatting issues
```

## Commands

```bash
# Basic linting
lint scripts/ models/

# Apply auto-fixes
lint scripts/ --fix

# Check formatting only
lint . --format-check

# Only show fixable issues
lint scripts/ --fix-only

# Include unsafe fixes
lint scripts/ --fix --unsafe-fixes

# Select specific rules
lint . --select E,F,W

# Ignore specific rules
lint . --ignore E501,W293

# Human-readable output
lint scripts/ --text

# Compact JSON (no indentation)
lint scripts/ --compact

# Exclude files
lint . --exclude "*/test_*.py" "*/__pycache__/*"

# Override max line length
lint . --max-line-length 120
```

## Violation Severities

| Severity | Description | Example Codes |
|----------|-------------|----------------|
| `error` | Actual errors that should be fixed | E, syntax errors |
| `warning` | Potential issues or style problems | W, E501 (line too long) |
| `lint` | Code quality issues (pyflakes) | F401 (unused import) |
| `info` | Informational messages | I (imports) |
| `refactor` | Code that could be simplified | N, R |
| `upgrade` | Python 2 to 3 upgrade suggestions | UP |
| `bugbear` | Likely bugs | B |
| `complexity` | Complex code | C |
| `security` | Security issues | S |

## Implementation Notes

The skill invokes Ruff via subprocess with JSON output parsing:
- `ruff check --output-format json` for linting
- `ruff format --check --diff` for formatting checks

Violations are categorized by severity based on Ruff error codes:
- E → error
- W → warning
- F → lint
- I → info
- N, R → refactor
- UP → upgrade
- B → bugbear
- C → complexity
- S → security

Fixable violations are tracked via the `fix` field in Ruff's JSON output.

When `--fix` is used, the skill applies auto-fixes and adds a warning indicating how many fixes were applied.

## Requirements

- Ruff must be installed: `pip install ruff` or `uv pip install ruff`
- Ruff must be available in PATH or accessible via `uv run ruff`

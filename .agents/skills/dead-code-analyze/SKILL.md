---
name: dead-code-analyze
description: Find and analyze dead Python code using vulture static analysis enhanced with Git history. Use when identifying unused functions, classes, imports, or unreachable code for cleanup or refactoring.
allowed-tools: Bash, Read
---

# Dead Code Analysis

Analyzes Python code to identify unused functions, classes, imports, and unreachable code using vulture static analysis, enhanced with Git history for context and risk assessment.

## Quick Start

```
dead-code-analyze models/ data/
dead-code-analyze models/ --min-confidence 90 --text
```

## Purpose

Use this skill when you want to:
- **Clean up code**: Identify unused code that can be safely removed
- **Find untested code**: Run on both library and test suite to find untested functions
- **Prep for refactoring**: Understand which code is actively used before major changes
- **Audit codebase**: Discover orphaned code that may indicate incomplete features
- **Reduce complexity**: Remove dead code to simplify maintenance

## Input Formats

### Direct paths
```
dead-code-analyze models/ data/
```
Analyze specific directories or files.

### Glob patterns
```
dead-code-analyze "**/*.py"
```
Use shell expansion to match multiple files.

### With exclusions
```
dead-code-analyze models/ --exclude "*/test_*.py" "*/__init__.py"
```
Exclude files matching patterns.

## Output Format

### JSON (default)
All skills now output a structured JSON envelope by default:

```json
{
  "ok": true,
  "name": "skill-name",
  "version": "1.0",
  "data": { /* skill-specific data */ },
  "warnings": [],
  "meta": {
    "latency_ms": 123.45,
    "timestamp": "2025-01-25T15:30:00Z",
    "input_path": "..."
  }
}
```

Use `--text` flag for human-readable output.

### Console output (--text)
```
================================================================================
DEAD CODE ANALYSIS REPORT
================================================================================
Analyzed: 23 files, 12,589 lines of code
Vulture findings: 47 items
Filtered to min-confidence 80: 27 items

HIGH (90-100%) - 9 items
--------------------------------------------------------------------------------
File:Line                        | Type        | Name                     | Size   | Last Modified  | Risk     
--------------------------------------------------------------------------------
./models/epiforecaster.py:234    | function    | _legacy_forward          | 42     | 2024-06-15     | LOW      
./data/cases_preprocessor.py:89  | method      | _deprecated_parse       | 18     | 2024-03-20     | LOW      
...

MEDIUM (70-89%) - 18 items
--------------------------------------------------------------------------------
...

RECOMMENDATIONS
--------------------------------------------------------------------------------
-  9 items: Safe to remove (high confidence, old code)
- 15 items: Investigate further (may be used dynamically)
-  3 items: Consider deprecation (add TODO/FIXME)

RISK SUMMARY
--------------------------------------------------------------------------------
- LOW     :  9 items (33.3%)
- MEDIUM  : 15 items (55.6%)
- HIGH    :  3 items (11.1%)
```

### JSON output (programmatic)
```
dead-code-analyze models/ --json
```
Returns structured JSON with summary and all findings:
```json
{
  "summary": {
    "total_files": 23,
    "total_lines": 12589,
    "total_findings": 47,
    "filtered_findings": 27,
    "findings_by_confidence": {"90%": 9, "80%": 15, "70%": 3},
    "findings_by_risk": {"LOW": 9, "MEDIUM": 15, "HIGH": 3},
    "recommendations": {"SAFE_TO_REMOVE": 9, "INVESTIGATE": 15, "CONSIDER_DEPRECATE": 3}
  },
  "findings": [
    {
      "filename": "models/epiforecaster.py",
      "line": 234,
      "name": "_legacy_forward",
      "kind": "function",
      "confidence": 100,
      "size": 42,
      "last_modified": "2024-06-15",
      "age_days": 224,
      "commit_count": 5,
      "risk_level": "LOW",
      "recommendation": "SAFE_TO_REMOVE"
    }
  ]
}
```

## Confidence Levels Explained

Vulture assigns confidence based on code type:

| Confidence | Code Types | Certainty |
|------------|------------|-----------|
| 100% | Function/class arguments, unreachable code | Certain won't execute |
| 90% | Imports | Almost certain |
| 60-89% | Attributes, classes, functions, methods, properties, variables | Rough estimate |

Use `--min-confidence` to filter:
- `100`: Only certain dead code (conservative)
- `80`: Balance of confidence and coverage (default)
- `60`: Find all potential dead code (more false positives)

## Risk Assessment Framework

Risk is assessed based on **confidence** and **Git history**:

| Risk Level | Criteria | Recommendation |
|------------|----------|----------------|
| LOW | 90-100% confidence, code ≥60 days old | **SAFE_TO_REMOVE** |
| MEDIUM | High confidence but recent, or medium confidence | **INVESTIGATE** or **CONSIDER_DEPRECATE** |
| HIGH | Low confidence (<70%), recent code | **INVESTIGATE_FURTHER** |

### Git history factors:
- **Age days**: Time since last modification
- **Commit count**: Number of commits touching that code
- **Threshold**: ≥60 days considered "old/stable"

## Typical Workflow

1. Run initial analysis
   ```bash
   dead-code-analyze models/ data/
   ```

2. Review findings by confidence and risk
   - Start with LOW risk items (safe to remove)
   - Investigate MEDIUM risk items (may be used dynamically)
   - Be cautious with HIGH risk items (need deeper analysis)

3. Generate whitelist for false positives
   ```bash
   dead-code-analyze models/ --make-whitelist > whitelist.py
   dead-code-analyze models/ whitelist.py
   ```

4. Remove verified dead code iteratively
   - Remove one batch at a time
   - Run tests after each batch
   - Re-run analysis to find more dead code

5. For production code, use conservative settings
   ```bash
   dead-code-analyze models/ --min-confidence 90 --sort-by-size
   ```

## Commands

```bash
# Basic analysis on directories
dead-code-analyze models/ data/

# Only high-confidence findings
dead-code-analyze models/ --min-confidence 90

# Sort by code size (largest dead code first)
dead-code-analyze models/ --sort-by-size

# Exclude test files and __init__.py
dead-code-analyze models/ --exclude "*/test_*.py" "*/__init__.py"

# Ignore specific patterns (e.g., test helpers)
dead-code-analyze models/ --ignore-names "test_*" "*_fixture"

# Ignore decorated functions (e.g., Flask routes)
dead-code-analyze models/ --ignore-decorators "@app.route" "@pytest.fixture"

# JSON output for programmatic use
dead-code-analyze models/ --json > findings.json

# Generate whitelist for all detected items
dead-code-analyze models/ --make-whitelist > whitelist.py

# Analyze specific files
dead-code-analyze models/epiforecaster.py data/epi_dataset.py
```

## False Positive Handling

Vulture can produce false positives due to:
- Dynamic code execution (`getattr`, `eval`, dynamic imports)
- Reflection and metaprogramming
- Test-only code accessed by test frameworks
- Decorated functions (e.g., Flask routes, pytest fixtures)

### Recommended strategies:

1. **Whitelist approach** (recommended)
   ```bash
   dead-code-analyze models/ --make-whitelist > whitelist.py
   ```
   Edit whitelist.py to keep only true false positives, then:
   ```bash
   dead-code-analyze models/ whitelist.py
   ```

2. **Exclusion patterns**
   - Exclude test files: `--exclude "*/test_*.py"`
   - Exclude vendor code: `--exclude "*/vendor/*"`

3. **Ignore patterns**
   - Ignore test helpers: `--ignore-names "test_*" "*_fixture"`
   - Ignore config access: `--ignore-names "cfg_*"`

4. **Ignore decorators**
   - Ignore Flask routes: `--ignore-decorators "@app.route"`
   - Ignore pytest fixtures: `--ignore-decorators "@pytest.fixture"`

## Limitations

- **Static analysis only**: Cannot detect code used via dynamic dispatch
- **May miss code**: Dynamically called code may be reported as unused
- **False positives**: Common in dynamic codebases, use whitelists
- **Git history required**: Risk assessment needs git repository

## Implementation Notes

The skill invokes `scripts/analysis/dead_code_analyze.py` which:
- Uses vulture's Python API (`vulture.Vulture().scavenge().get_unused_code()`)
- Extracts Item attributes: filename, line, name, kind, confidence, size
- Enhances with Git history using `git log` for each finding
- Assesses risk based on confidence (60-100%) and code age (≥60 days old)
- Provides actionable recommendations: SAFE_TO_REMOVE, INVESTIGATE, CONSIDER_DEPRECATE
- Supports JSON output for programmatic analysis
- Can generate whitelists automatically

Key thresholds:
- Default min-confidence: 80 (balance of coverage and accuracy)
- Old code threshold: ≥60 days
- Risk classification: LOW/MEDIUM/HIGH based on confidence × age

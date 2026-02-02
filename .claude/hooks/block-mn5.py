#!/usr/bin/env python3
"""
Claude Code Hook: Block MN5 Production Configs
==============================================
Blocks training/preprocessing with MN5 configs that require cluster resources.
"""
import json
import re
import sys

# Patterns that indicate MN5 production configs
_BLOCKED_PATTERNS = [
    r"configs/production_only/",
    r".*mn5.*\.yaml",
    r"train_epifor_mn5_",
    r"preprocess_mn5_",
]

def _check_blocked(command: str) -> str | None:
    """Check if command matches blocked patterns. Returns error message if blocked."""
    # Allow remote-train and remote-preprocess - they submit to cluster, don't run locally
    if re.search(r"remote-(train|preprocess)\s+|remote_(train|preprocess)\.py", command):
        return None

    # Only block actual training/preprocessing commands (python/uv run)
    # Allow git, editor, and other safe operations on these files
    training_prefixes = [
        r"^\s*uv\s+run\s+",
        r"^\s*python\s+",
        r"^\s*python3\s+",
        r"^\s*sbatch\s+",
    ]

    has_training_prefix = any(
        re.search(prefix, command) for prefix in training_prefixes
    )

    if not has_training_prefix:
        return None  # Allow non-training commands

    # Check if it uses MN5 production configs
    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, command):
            return (
                "BLOCKED: MN5 Production Config Detected\n\n"
                "This command requires MN5 cluster resources (100GB+ RAM, GPU) and\n"
                "should NOT be run on your local machine.\n\n"
                f"Command: {command}\n\n"
                "Why this is blocked:\n"
                "- Memory requirement: 100GB+ (your machine: ~16GB)\n"
                "- Intended environment: MN5 cluster with SLURM\n"
                "- OOM risk: Very high if run locally\n\n"
                "Alternatives:\n"
                "1. Use a development config instead:\n"
                "   uv run train epiforecaster --config configs/train_epifor_full.yaml\n\n"
                "2. For production runs, submit via SLURM:\n"
                "   sbatch scripts/train_single_gpu.sbatch\n\n"
                "Documentation: configs/production_only/README.md"
            )
    return None

def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    tool_name = input_data.get("tool_name", "")
    if tool_name != "Bash":
        sys.exit(0)

    tool_input = input_data.get("tool_input", {})
    command = tool_input.get("command", "")

    if not command:
        sys.exit(0)

    error_message = _check_blocked(command)
    if error_message:
        print(error_message, file=sys.stderr)
        # Exit code 2 blocks tool call and shows stderr to Claude
        sys.exit(2)

    sys.exit(0)

if __name__ == "__main__":
    main()

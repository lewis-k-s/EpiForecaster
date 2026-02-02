#!/usr/bin/env python3
"""CLI for dry-run testing of sbatch script generation.

This tool allows you to preview generated sbatch scripts before submitting
them to the cluster. Useful for debugging and template development.

Examples:
    # Preview training script
    uv run python scripts/generate_sbatch_preview.py training \
        --config configs/train.yaml \
        --override training.batch_size=32 \
        --override training.epochs=5

    # Preview optuna HPO script
    uv run python scripts/generate_sbatch_preview.py optuna \
        --study-name test_study \
        --config configs/train.yaml \
        --array-size 4

    # Save to file
    uv run python scripts/generate_sbatch_preview.py training \
        --config configs/train.yaml \
        --output /tmp/preview.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.remote_runner import (
    SlurmJobSpec,
    generate_optuna_script,
    generate_training_script,
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate and preview sbatch scripts for MN5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s training --config configs/train.yaml
  %(prog)s training --config configs/train.yaml -o training.batch_size=32
  %(prog)s optuna --study-name hpo_v1 --config configs/train.yaml --array-size 8
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Training command
    train_parser = subparsers.add_parser(
        "training",
        help="Generate training script",
    )
    train_parser.add_argument(
        "--config",
        required=True,
        help="Path to training config YAML",
    )
    train_parser.add_argument(
        "--job-name",
        default="training_preview",
        help="Job name (default: training_preview)",
    )
    train_parser.add_argument(
        "--time",
        default="04:00:00",
        help="Time limit (default: 04:00:00)",
    )
    train_parser.add_argument(
        "--gres",
        default="gpu:1",
        help="GRES specification (default: gpu:1)",
    )
    train_parser.add_argument(
        "-o",
        "--override",
        action="append",
        dest="overrides",
        default=[],
        help="Config override (can be used multiple times)",
    )
    train_parser.add_argument(
        "--output",
        "-O",
        help="Output file path (default: print to stdout)",
    )

    # Optuna command
    optuna_parser = subparsers.add_parser(
        "optuna",
        help="Generate Optuna HPO script",
    )
    optuna_parser.add_argument(
        "--study-name",
        required=True,
        help="Name of the Optuna study",
    )
    optuna_parser.add_argument(
        "--config",
        required=True,
        help="Path to training config YAML",
    )
    optuna_parser.add_argument(
        "--job-name",
        help="Job name (default: same as study-name)",
    )
    optuna_parser.add_argument(
        "--array-size",
        type=int,
        default=4,
        help="Number of array tasks (default: 4)",
    )
    optuna_parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Epochs per trial (default: 2)",
    )
    optuna_parser.add_argument(
        "--time",
        default="04:00:00",
        help="Time limit (default: 04:00:00)",
    )
    optuna_parser.add_argument(
        "--output",
        "-O",
        help="Output file path (default: print to stdout)",
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "training":
        # Generate training script
        job_spec = SlurmJobSpec(
            job_name=args.job_name,
            time=args.time,
            gres=args.gres,
        )
        commands = generate_training_script(
            config_path=args.config,
            overrides=args.overrides if args.overrides else None,
        )
        script = job_spec.to_sbatch_script(commands)

    elif args.command == "optuna":
        # Generate optuna script
        job_name = args.job_name or args.study_name
        job_spec = SlurmJobSpec(
            job_name=job_name,
            time=args.time,
            gres="gpu:1",
            array=f"0-{args.array_size - 1}",
        )
        commands = generate_optuna_script(
            study_name=args.study_name,
            config_path=args.config,
            epochs=args.epochs,
        )
        script = job_spec.to_sbatch_script(commands)

    else:
        parser.print_help()
        sys.exit(1)

    # Output script
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(script)
        print(f"Script written to: {output_path}")

        # Validate with bash -n
        import subprocess

        result = subprocess.run(
            ["bash", "-n", str(output_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✓ Script passes bash syntax check")
        else:
            print(f"✗ Script has syntax errors:\n{result.stderr}")
            sys.exit(1)
    else:
        # Print to stdout
        print("=" * 70)
        print("Generated SBATCH Script")
        print("=" * 70)
        print(script)
        print("=" * 70)

        # Validate
        import subprocess

        result = subprocess.run(
            ["bash", "-n"],
            input=script,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✓ Script passes bash syntax check")
        else:
            print(f"✗ Script has syntax errors:\n{result.stderr}")
            sys.exit(1)


if __name__ == "__main__":
    main()

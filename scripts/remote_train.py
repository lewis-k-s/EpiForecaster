#!/usr/bin/env python3
"""Submit training jobs to MN5 cluster via SSH.

This script provides a CLI wrapper around utils.remote_runner for submitting
training and Optuna HPO jobs to the MareNostrum 5 cluster.
"""

import argparse

from scripts.utils.skill_output import SkillOutputBuilder, print_output
from utils.remote_runner import run_optuna_study, run_training, wait_until_running


def main():
    parser = argparse.ArgumentParser(
        description="Submit training jobs to MN5 cluster via SSH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (wait until RUNNING, then return)
  remote-train my_experiment configs/train.yaml

  # With time limit and overrides
  remote-train batch_test configs/train.yaml --time 02:00:00 --override training.batch_size=48

  # Optuna HPO
  remote-train hpo_v3 configs/train.yaml --optuna --array-size 8 --epochs 3

  # Submit and return immediately (no waiting)
  remote-train my_experiment configs/train.yaml --no-wait
        """,
    )
    parser.add_argument(
        "experiment_name",
        help="Name for tracking the experiment (also used as study name for Optuna)",
    )
    parser.add_argument(
        "config_path",
        help="Path to training config YAML (on remote)",
    )

    # SLURM options
    parser.add_argument(
        "--time",
        default="01:00:00",
        help="SLURM time limit in HH:MM:SS format (default: 01:00:00)",
    )
    parser.add_argument(
        "--gres",
        default="gpu:1",
        help="SLURM gres specification (default: gpu:1)",
    )

    # Config options
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        help="Config overrides (e.g., training.batch_size=48). Can be repeated.",
    )

    # Optuna options
    parser.add_argument(
        "--optuna",
        action="store_true",
        help="Run Optuna HPO instead of single training job",
    )
    parser.add_argument(
        "--array-size",
        type=int,
        default=4,
        help="Number of Optuna parallel workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Epochs per Optuna trial (default: 2)",
    )

    # Behavior options
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip syncing code to remote first",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit job and return immediately (default: wait until RUNNING + 30s for initial logs)",
    )

    args = parser.parse_args()

    builder = SkillOutputBuilder(
        skill_name="remote-train",
        input_path=args.config_path,
    )

    # Build kwargs for remote_runner functions
    kwargs = {
        "skip_sync_to": args.no_sync,
        "timeout_secs": 0 if args.no_wait else 60,  # 0 = no polling
        "poll_interval_secs": 10,
    }

    try:
        if args.optuna:
            # Run Optuna HPO
            result = run_optuna_study(
                study_name=args.experiment_name,
                config_path=args.config_path,
                epochs=args.epochs,
                array_size=args.array_size,
                time=args.time,
                **kwargs,
            )
        else:
            # Run single training job
            result = run_training(
                experiment_name=args.experiment_name,
                config_path=args.config_path,
                overrides=args.overrides,
                time=args.time,
                gres=args.gres,
                **kwargs,
            )

        # Wait until job is RUNNING (unless --no-wait is set)
        initial_logs = ""
        if not args.no_wait and result.job_id:
            print(f"Waiting for job {result.job_id} to start...")
            final_status, initial_logs = wait_until_running(result.job_id)
            result.status = final_status

        # Build output data
        output_data = {
            "job_id": result.job_id,
            "experiment_name": result.experiment_name,
            "status": result.status.value,
        }

        if result.local_output_path:
            output_data["local_output_path"] = str(result.local_output_path)

        if result.error_message:
            output_data["error_message"] = result.error_message

        if initial_logs:
            output_data["initial_logs"] = initial_logs

        # Add metadata about wait mode
        builder.add_meta("no_wait", args.no_wait)

        output = builder.success(output_data)
        print_output(output)

    except FileNotFoundError:
        print_output(
            builder.error(
                "FileNotFoundError",
                f"Config file not found: {args.config_path}",
            )
        )
    except Exception as e:
        print_output(
            builder.error(
                type(e).__name__,
                str(e),
                {"config_path": args.config_path, "experiment": args.experiment_name},
            )
        )


if __name__ == "__main__":
    main()

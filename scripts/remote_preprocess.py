#!/usr/bin/env python3
"""Submit preprocessing jobs to MN5 cluster via SSH.

This script provides a CLI wrapper around utils.remote_runner for submitting
preprocessing jobs to the MareNostrum 5 cluster.
"""

import argparse

from scripts.utils.skill_output import SkillOutputBuilder, print_output
from utils.remote_runner import run_preprocessing, wait_until_running


def main():
    parser = argparse.ArgumentParser(
        description="Submit preprocessing jobs to MN5 cluster via SSH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing (wait until RUNNING, then return)
  remote-preprocess synth_preprocess configs/preprocess_mn5_synth.yaml

  # With more CPUs and shorter time
  remote-preprocess fast_preprocess configs/preprocess.yaml --cpus-per-task 80 --time 01:00:00

  # Submit and return immediately (no waiting)
  remote-preprocess my_preprocess configs/preprocess.yaml --no-wait
        """,
    )
    parser.add_argument(
        "experiment_name",
        help="Name for tracking the preprocessing job",
    )
    parser.add_argument(
        "config_path",
        help="Path to preprocessing config YAML (on remote)",
    )

    # SLURM options
    parser.add_argument(
        "--time",
        default="04:00:00",
        help="SLURM time limit in HH:MM:SS format (default: 04:00:00)",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=40,
        help="Number of CPUs for parallel processing (default: 40)",
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
        skill_name="remote-preprocess",
        input_path=args.config_path,
    )

    # Build kwargs for run_preprocessing
    kwargs = {
        "skip_sync_to": args.no_sync,
        "timeout_secs": 0 if args.no_wait else 60,  # 0 = no polling
        "poll_interval_secs": 10,
    }

    try:
        # Run preprocessing job
        result = run_preprocessing(
            experiment_name=args.experiment_name,
            config_path=args.config_path,
            time=args.time,
            cpus_per_task=args.cpus_per_task,
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

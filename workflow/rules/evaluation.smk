"""Checkpoint evaluation with declared machine-readable evidence."""


if WORKFLOW_KIND in {"train", "evaluate", "hpo"}:
    rule evaluate_model:
        input:
            checkpoint=str(EVALUATION_CHECKPOINT),
            prerequisites=EVALUATION_PREREQUISITES,
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["evaluation"]),
        output:
            artifacts=EVALUATION_ARTIFACTS,
            manifest=str(EVALUATION_MANIFEST),
            marker=str(EVALUATION_DONE),
        threads:
            CPUS_PER_TASK
        resources:
            gpu=1
        log:
            str(RUN_ROOT / "logs/evaluate_model.log")
        params:
            command=slurm_step(
                container_command(
                    [
                        "python",
                        "-m",
                        "scripts.workflow.run_support",
                        "evaluate",
                        "--config",
                        CONTAINER_CONFIG,
                        "--run-root",
                        CONTAINER_RUN,
                        "--checkpoint",
                        EVALUATION_CONTAINER_CHECKPOINT,
                    ],
                    run_root=RUN_ROOT,
                    source_root=SOURCE_ROOT,
                    sif=SIF,
                    dvc_remote=DVC_REMOTE,
                    dvc_cache=DVC_CACHE,
                    stage_data=True,
                    nv=True,
                    cpus=CPUS_PER_TASK,
                    input_paths=EVALUATION_INPUT_PATHS,
                ),
                CPUS_PER_TASK,
            )
        shell:
            "{params.command} > {log:q} 2>&1"

"""Single-model training and its explicit checkpoint contract."""


if WORKFLOW_KIND == "train":
    rule train_model:
        input:
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["training"]),
        output:
            checkpoint=str(TRAINING_CHECKPOINT),
            manifest=str(TRAINING_MANIFEST),
            marker=str(TRAINING_DONE),
        threads:
            CPUS_PER_TASK
        log:
            str(RUN_ROOT / "logs/train_model.log")
        params:
            command=slurm_step(
                container_command(
                    [
                        "python",
                        "-m",
                        "scripts.workflow.run_support",
                        "train",
                        "--config",
                        CONTAINER_CONFIG,
                        "--source-root",
                        CONTAINER_SOURCE,
                        "--run-root",
                        CONTAINER_RUN,
                    ],
                    run_root=RUN_ROOT,
                    source_root=SOURCE_ROOT,
                    sif=SIF,
                    dvc_remote=DVC_REMOTE,
                    dvc_cache=DVC_CACHE,
                    stage_data=True,
                    nv=True,
                    cpus=CPUS_PER_TASK,
                ),
                CPUS_PER_TASK,
            )
        shell:
            "{params.command} > {log:q} 2>&1"

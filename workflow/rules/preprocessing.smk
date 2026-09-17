"""Immutable preprocessing and DVC publication for raw inputs."""


if WORKFLOW_KIND == "preprocess":
    rule preprocess_dataset:
        input:
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["preprocessing"]),
        output:
            dataset=str(PREPROCESSING_DATASET),
            config=str(PREPROCESSING_EFFECTIVE_CONFIG),
            qa=str(PREPROCESSING_QA),
            marker=str(PREPROCESSING_DONE),
        threads:
            CPUS_PER_TASK
        log:
            str(RUN_ROOT / "logs/preprocess_dataset.log")
        params:
            command=slurm_step(
                container_command(
                    [
                        "python",
                        "-m",
                        "scripts.workflow.run_support",
                        "preprocess",
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
                    cpus=CPUS_PER_TASK,
                ),
                CPUS_PER_TASK,
                gpu=False,
            )
        shell:
            "{params.command} > {log:q} 2>&1"


    rule publish_preprocessed:
        input:
            dataset=str(PREPROCESSING_DATASET),
            marker=str(PREPROCESSING_DONE),
            qa=str(PREPROCESSING_QA),
        output:
            pointer=str(PREPROCESSING_POINTER),
            receipt=str(PREPROCESSING_RECEIPT),
            marker=str(PREPROCESSING_PUBLISHED),
        log:
            str(RUN_ROOT / "logs/publish_preprocessed.log")
        params:
            command=container_command(
                [
                    "python",
                    "-m",
                    "scripts.workflow.run_support",
                    "publish-preprocessed",
                    "--config",
                    CONTAINER_CONFIG,
                    "--run-root",
                    CONTAINER_RUN,
                ],
                run_root=RUN_ROOT,
                source_root=SOURCE_ROOT,
                sif=SIF,
                dvc_remote=DVC_REMOTE,
                dvc_cache=DVC_CACHE,
                dvc_mode="push",
                cpus=CPUS_PER_TASK,
                promotion_root=str(PREPROCESSING_ROOT),
            )
        shell:
            "{params.command} > {log:q} 2>&1"

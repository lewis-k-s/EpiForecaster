"""HPO promotion and post-promotion checkpoint cleanup."""


if WORKFLOW_KIND == "hpo":
    rule promote_best_model:
        input:
            selection=str(SELECTION),
            evaluation=str(EVALUATION_MANIFEST),
            evaluation_artifacts=EVALUATION_ARTIFACTS,
            manifest=str(MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["promotion"]),
        output:
            str(PROMOTED)
        log:
            str(RUN_ROOT / "logs/promote_best_model.log")
        params:
            command=container_command(
                [
                    "python",
                    "-m",
                    "scripts.workflow.run_support",
                    "promote",
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
                promotion_root=PROMOTION_ROOT,
            )
        shell:
            "{params.command} > {log:q} 2>&1"

    rule cleanup_losing_checkpoints:
        input:
            promoted=str(PROMOTED),
            selection=str(SELECTION),
        output:
            str(CHECKPOINTS_CLEANED)
        log:
            str(RUN_ROOT / "logs/cleanup_losing_checkpoints.log")
        params:
            command=container_command(
                [
                    "python",
                    "-m",
                    "scripts.workflow.run_support",
                    "cleanup-study",
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
                cpus=CPUS_PER_TASK,
            )
        shell:
            "{params.command} > {log:q} 2>&1"

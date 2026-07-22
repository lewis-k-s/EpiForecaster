"""Concurrent HPO and ablation study execution plus deterministic reduction."""


if WORKFLOW_KIND == "hpo":
    rule launch_optuna_workers:
        input:
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["hpo"]),
        output:
            journal=str(JOURNAL),
            workers=expand(
                str(RUN_ROOT / "optuna/workers/{worker_id}.done"),
                worker_id=WORKER_IDS,
            ),
            complete=str(WORKERS_COMPLETE),
        threads:
            STUDY_WORKERS * CPUS_PER_TASK
        log:
            str(RUN_ROOT / "logs/optuna-workers.log")
        params:
            command=launch_workers_command(
                hpo_worker_arguments(config),
                run_root=RUN_ROOT,
                source_root=SOURCE_ROOT,
                sif=SIF,
                workers=STUDY_WORKERS,
                cpus_per_task=CPUS_PER_TASK,
            )
        shell:
            "{params.command} > {log:q} 2>&1 && touch {output.complete:q}"


    rule select_best_trial:
        input:
            workers=str(WORKERS_COMPLETE),
            journal=str(JOURNAL),
            manifest=str(MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["selection"]),
        output:
            summary=str(STUDY_SUMMARY),
            best_config=str(BEST_CONFIG),
            checkpoint=str(BEST_MODEL),
            selection=str(SELECTION),
        log:
            str(RUN_ROOT / "logs/select_best_trial.log")
        params:
            command=container_command(
                [
                    "python",
                    "-m",
                    "scripts.workflow.run_support",
                    "select-best",
                    "--config",
                    CONTAINER_CONFIG,
                    "--run-root",
                    CONTAINER_RUN,
                    "--journal",
                    CONTAINER_JOURNAL,
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

elif WORKFLOW_KIND == "ablation":
    rule launch_ablation_workers:
        input:
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["ablation"]),
        output:
            journal=str(JOURNAL),
            workers=expand(
                str(RUN_ROOT / "optuna/workers/{worker_id}.done"),
                worker_id=WORKER_IDS,
            ),
            complete=str(WORKERS_COMPLETE),
        threads:
            STUDY_WORKERS * CPUS_PER_TASK
        log:
            str(RUN_ROOT / "logs/ablation-workers.log")
        params:
            command=launch_workers_command(
                ablation_worker_arguments(config),
                run_root=RUN_ROOT,
                source_root=SOURCE_ROOT,
                sif=SIF,
                workers=STUDY_WORKERS,
                cpus_per_task=CPUS_PER_TASK,
            )
        shell:
            "{params.command} > {log:q} 2>&1 && touch {output.complete:q}"


    rule finalize_ablation:
        input:
            workers=str(WORKERS_COMPLETE),
            journal=str(JOURNAL),
            manifest=str(MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["ablation"]),
        output:
            required=[str(path) for path in ABLATION_REQUIRED],
            manifest=str(ABLATION_MANIFEST),
            marker=str(ABLATION_DONE),
        log:
            str(RUN_ROOT / "logs/finalize_ablation.log")
        params:
            command=container_command(
                [
                    "python",
                    "-m",
                    "scripts.workflow.run_support",
                    "finalize-ablation",
                    "--config",
                    CONTAINER_CONFIG,
                    "--run-root",
                    CONTAINER_RUN,
                    "--journal",
                    CONTAINER_JOURNAL,
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

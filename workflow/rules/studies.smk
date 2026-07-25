"""Concurrent HPO and ablation study execution plus deterministic reduction."""


if WORKFLOW_KIND == "hpo":
    rule launch_optuna_workers:
        input:
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["hpo"]),
        output:
            journal=str(JOURNAL),
            workers=expand(
                str(RUN_ROOT / ".state/workers/hpo/{worker_id}.done"),
                worker_id=WORKER_IDS,
            ),
            complete=str(WORKERS_COMPLETE),
        threads:
            STUDY_WORKERS * CPUS_PER_TASK
        log:
            str(RUN_ROOT / "logs/optuna-workers.log")
        params:
            command=launch_parallel_workers_command(
                hpo_worker_arguments(config),
                run_root=RUN_ROOT,
                source_root=SOURCE_ROOT,
                sif=SIF,
                workers=STUDY_WORKERS,
                cpus_per_task=CPUS_PER_TASK,
                worker_group="hpo",
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
                str(RUN_ROOT / ".state/workers/ablation/{worker_id}.done"),
                worker_id=WORKER_IDS,
            ),
            complete=str(WORKERS_COMPLETE),
        threads:
            STUDY_WORKERS * CPUS_PER_TASK
        log:
            str(RUN_ROOT / "logs/ablation-workers.log")
        params:
            command=launch_parallel_workers_command(
                ablation_worker_arguments(config),
                run_root=RUN_ROOT,
                source_root=SOURCE_ROOT,
                sif=SIF,
                workers=STUDY_WORKERS,
                cpus_per_task=CPUS_PER_TASK,
                worker_group="ablation",
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

elif WORKFLOW_KIND == "crossval":
    rule train_crossval_folds:
        input:
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["crossval"]),
        output:
            workers=expand(
                str(RUN_ROOT / ".state/workers/crossval-training/{worker_id}.done"),
                worker_id=WORKER_IDS,
            ),
            complete=str(CROSSVAL_TRAINING_COMPLETE),
        threads:
            TASKS * CPUS_PER_TASK
        log:
            str(RUN_ROOT / "logs/crossval-training-workers.log")
        params:
            command=lambda wildcards: launch_parallel_workers_command(
                crossval_worker_arguments("training"),
                run_root=RUN_ROOT,
                source_root=SOURCE_ROOT,
                sif=SIF,
                workers=STUDY_WORKERS,
                cpus_per_task=CPUS_PER_TASK,
                worker_group="crossval-training",
            )
        shell:
            "{params.command} > {log:q} 2>&1 && touch {output.complete:q}"


    rule evaluate_crossval_folds:
        input:
            training=str(CROSSVAL_TRAINING_COMPLETE),
            dependencies=str(DEPENDENCY_CONTRACTS["evaluation"]),
        output:
            workers=expand(
                str(RUN_ROOT / ".state/workers/crossval-evaluation/{worker_id}.done"),
                worker_id=WORKER_IDS,
            ),
            complete=str(WORKERS_COMPLETE),
        threads:
            TASKS * CPUS_PER_TASK
        log:
            str(RUN_ROOT / "logs/crossval-evaluation-workers.log")
        params:
            command=lambda wildcards: launch_parallel_workers_command(
                crossval_worker_arguments("evaluation"),
                run_root=RUN_ROOT,
                source_root=SOURCE_ROOT,
                sif=SIF,
                workers=STUDY_WORKERS,
                cpus_per_task=CPUS_PER_TASK,
                worker_group="crossval-evaluation",
            )
        shell:
            "{params.command} > {log:q} 2>&1 && touch {output.complete:q}"


    rule finalize_crossval:
        input:
            workers=str(WORKERS_COMPLETE),
            dependencies=str(DEPENDENCY_CONTRACTS["evaluation"]),
        output:
            fold_metrics=str(CROSSVAL_FOLD_METRICS),
            aggregate_metrics=str(CROSSVAL_AGGREGATE_METRICS),
            manifest=str(CROSSVAL_MANIFEST),
            marker=str(CROSSVAL_DONE),
        log:
            str(RUN_ROOT / "logs/finalize_crossval.log")
        params:
            command=container_command(
                [
                    "python",
                    "-m",
                    "scripts.workflow.run_support",
                    "finalize-crossval",
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

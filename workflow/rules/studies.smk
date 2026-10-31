"""Concurrent HPO and ablation study execution plus deterministic reduction."""


if WORKFLOW_KIND in {"hpo", "ablation"}:
    # Only initialization owns the journal as an output. Worker jobs mutate
    # this durable input and own a disposable completion receipt instead.
    rule initialize_study:
        output:
            str(JOURNAL)
        shell:
            "touch {output:q}"


if WORKFLOW_KIND == "hpo":
    rule launch_optuna_workers:
        input:
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["hpo"]),
            journal=str(JOURNAL),
        output:
            complete=str(WORKERS_COMPLETE),
        threads:
            STUDY_WORKERS * CPUS_PER_TASK
        resources:
            gpu=STUDY_WORKERS
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
                host_step=True,
            )
        shell:
            "{params.command} > {log:q} 2>&1"

elif WORKFLOW_KIND == "ablation":
    rule launch_ablation_workers:
        input:
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["ablation"]),
            journal=str(JOURNAL),
        output:
            complete=str(WORKERS_COMPLETE),
        threads:
            STUDY_WORKERS * CPUS_PER_TASK
        resources:
            gpu=STUDY_WORKERS
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
                input_paths=(
                    [str(config["ablation"]["reuse_evidence_dir"])]
                    if config["ablation"].get("reuse_evidence_dir") is not None
                    else []
                ),
                host_step=True,
                stage_data=True,
            )
        shell:
            "{params.command} > {log:q} 2>&1"

elif WORKFLOW_KIND == "ablation_suite":
    rule initialize_suite_study:
        output:
            str(RUN_ROOT / "optuna/{study}.journal")
        wildcard_constraints:
            study="|".join(SUITE_NAMES)
        shell:
            "touch {output:q}"


    rule launch_suite_study_workers:
        input:
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["ablation"]),
            journal=str(RUN_ROOT / "optuna/{study}.journal"),
            previous=lambda wildcards: suite_previous_manifest(wildcards.study),
        output:
            str(RUN_ROOT / ".state/ablation_suite/{study}.workers.done")
        wildcard_constraints:
            study="|".join(SUITE_NAMES)
        threads:
            4 * CPUS_PER_TASK
        resources:
            gpu=4
        log:
            str(RUN_ROOT / "logs/ablation-suite-{study}-workers.log")
        params:
            command=lambda wildcards: launch_parallel_workers_command(
                suite_worker_arguments(wildcards.study),
                run_root=RUN_ROOT,
                source_root=SOURCE_ROOT,
                sif=SIF,
                workers=4,
                cpus_per_task=CPUS_PER_TASK,
                worker_group=f"ablation_{wildcards.study}",
            )
        shell:
            "{params.command} > {log:q} 2>&1 && touch {output:q}"


    rule finalize_suite_study:
        input:
            workers=str(RUN_ROOT / ".state/ablation_suite/{study}.workers.done"),
            journal=str(RUN_ROOT / "optuna/{study}.journal"),
            manifest=str(MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["ablation"]),
        output:
            manifest=str(SUITE_ROOT / "{study}/ablation_manifest.json"),
            marker=str(RUN_ROOT / ".state/ablation_suite/{study}.done"),
        wildcard_constraints:
            study="|".join(SUITE_NAMES)
        log:
            str(RUN_ROOT / "logs/ablation-suite-{study}-finalize.log")
        params:
            command=lambda wildcards: container_command(
                [
                    "python", "-m", "scripts.workflow.run_support", "finalize-ablation",
                    "--config", CONTAINER_CONFIG,
                    "--run-root", CONTAINER_RUN,
                    "--journal", CONTAINER_RUN / "optuna" / f"{wildcards.study}.journal",
                    "--study-name", wildcards.study,
                ],
                run_root=RUN_ROOT,
                source_root=SOURCE_ROOT,
                sif=SIF,
                dvc_remote=DVC_REMOTE,
                dvc_cache=DVC_CACHE,
                cpus=CPUS_PER_TASK,
                host_step=True,
                stage_data=True,
            )
        shell:
            "{params.command} > {log:q} 2>&1"


    rule finalize_ablation_suite:
        input:
            [str(SUITE_ROOT / name / "ablation_manifest.json") for name in SUITE_NAMES]
        output:
            str(SUITE_MANIFEST)
        log:
            str(RUN_ROOT / "logs/finalize-ablation-suite.log")
        shell:
            "PYTHONPATH={SOURCE_ROOT:q} {HOST_PYTHON:q} -m scripts.workflow.run_support "
            "finalize-ablation-suite --config {CONFIG_PATH:q} --run-root {RUN_ROOT:q} "
            "> {log:q} 2>&1"

elif WORKFLOW_KIND == "crossval":
    rule train_crossval_fold:
        input:
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["crossval"]),
        output:
            checkpoint=str(
                CROSSVAL_FOLD_ROOT / "outputs/training/training/model/checkpoints/best_model.pt"
            ),
            training=str(
                CROSSVAL_FOLD_ROOT / "outputs/training/training/model/training_manifest.json"
            ),
            config=str(CROSSVAL_FOLD_ROOT / "config.yaml"),
            manifest=str(CROSSVAL_FOLD_ROOT / "fold_training_manifest.json"),
        wildcard_constraints:
            fold="|".join(WORKER_IDS)
        threads:
            CPUS_PER_TASK
        resources:
            gpu=1
        log:
            str(RUN_ROOT / "logs/crossval-training-{fold}.log")
        params:
            command=lambda wildcards: crossval_fold_command("training", wildcards.fold)
        shell:
            "{params.command} > {log:q} 2>&1"


    rule evaluate_crossval_fold:
        input:
            checkpoint=str(
                CROSSVAL_FOLD_ROOT / "outputs/training/training/model/checkpoints/best_model.pt"
            ),
            training=str(
                CROSSVAL_FOLD_ROOT / "outputs/training/training/model/training_manifest.json"
            ),
            config=str(CROSSVAL_FOLD_ROOT / "config.yaml"),
            manifest=str(CROSSVAL_FOLD_ROOT / "fold_training_manifest.json"),
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["evaluation"]),
        output:
            artifacts=CROSSVAL_EVALUATION_ARTIFACTS,
            evaluation=str(
                CROSSVAL_FOLD_ROOT / "outputs/evaluation/evaluation_manifest.json"
            ),
            manifest=str(CROSSVAL_FOLD_ROOT / "fold_manifest.json"),
        wildcard_constraints:
            fold="|".join(WORKER_IDS)
        threads:
            CPUS_PER_TASK
        resources:
            gpu=1
        log:
            str(RUN_ROOT / "logs/crossval-evaluation-{fold}.log")
        params:
            command=lambda wildcards: crossval_fold_command("evaluation", wildcards.fold)
        shell:
            "{params.command} > {log:q} 2>&1"


    rule finalize_crossval:
        input:
            manifests=expand(
                str(CROSSVAL_FOLD_ROOT / "fold_manifest.json"),
                fold=WORKER_IDS,
            ),
            evaluations=expand(
                str(CROSSVAL_FOLD_ROOT / "outputs/evaluation/evaluation_manifest.json"),
                fold=WORKER_IDS,
            ),
            artifacts=expand(CROSSVAL_EVALUATION_ARTIFACTS, fold=WORKER_IDS),
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
                host_step=True,
            )
        shell:
            "{params.command} > {log:q} 2>&1"

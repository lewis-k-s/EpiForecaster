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


    rule export_training_provenance:
        input:
            training=str(TRAINING_MANIFEST),
            run_manifest=str(MANIFEST),
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["training"]),
        output:
            str(TRAINING_PROVENANCE)
        log:
            str(RUN_ROOT / "logs/export_training_provenance.log")
        shell:
            "PYTHONPATH={SOURCE_ROOT:q} {HOST_PYTHON:q} "
            "-m scripts.analysis.query_run_provenance "
            "{TRAINING_ROOT:q} --run-manifest {input.run_manifest:q} "
            "--output {output:q} --compact > {log:q} 2>&1"


    if TRAINING_EVIDENCE_RETENTION == "dvc":
        rule retain_training_evidence:
            input:
                provenance=str(TRAINING_PROVENANCE),
                training=str(TRAINING_MANIFEST),
            output:
                str(TRAINING_EVIDENCE_RECORD)
            log:
                str(RUN_ROOT / "logs/retain_training_evidence.log")
            params:
                command=container_command(
                    [
                        "python",
                        "-m",
                        "scripts.workflow.run_support",
                        "retain-training-evidence",
                        "--config",
                        CONTAINER_CONFIG,
                        "--run-root",
                        CONTAINER_RUN,
                        "--model-root",
                        CONTAINER_RUN / TRAINING_ROOT.relative_to(RUN_ROOT),
                    ],
                    run_root=RUN_ROOT,
                    source_root=SOURCE_ROOT,
                    sif=SIF,
                    dvc_remote=DVC_REMOTE,
                    dvc_cache=DVC_CACHE,
                    dvc_mode="push",
                    cpus=CPUS_PER_TASK,
                    promotion_root=str(RUN_ROOT / "evidence"),
                )
            shell:
                "{params.command} > {log:q} 2>&1"

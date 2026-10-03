"""Single-model training and its explicit checkpoint contract."""


if WORKFLOW_KIND == "train":
    rule native_source_sets:
        input:
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["training"]),
        output:
            manifest=str(RUN_ROOT / "inputs/native_source_sets/manifest.json"),
            assignments=str(RUN_ROOT / "inputs/native_source_sets/assignments.parquet"),
            receipt=str(RUN_ROOT / ".state/native_splits.json"),
        threads: 1
        log: str(RUN_ROOT / "logs/native_source_sets.log")
        params:
            command=container_command(
                ["python", "-m", "scripts.workflow.run_support", "native-splits",
                 "--config", CONTAINER_CONFIG, "--source-root", CONTAINER_SOURCE, "--run-root", CONTAINER_RUN],
                run_root=RUN_ROOT, source_root=SOURCE_ROOT, sif=SIF,
                dvc_remote=DVC_REMOTE, dvc_cache=DVC_CACHE, stage_data=True,
                dvc_mode="push", cpus=1, host_step=True,
            )
        shell: "{params.command} > {log:q} 2>&1"

    rule train_model:
        input:
            splits=str(RUN_ROOT / ".state/native_splits.json"),
            data=str(DATA_MANIFEST),
            dependencies=str(DEPENDENCY_CONTRACTS["training"]),
        output:
            checkpoint=str(TRAINING_CHECKPOINT),
            manifest=str(TRAINING_MANIFEST),
            dynamics_manifest=str(TRAINING_DYNAMICS_MANIFEST),
            dynamics=str(TRAINING_DYNAMICS),
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
                job_name="train-native",
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



if WORKFLOW_KIND in {"train", "evaluate"} and TRAINING_EVIDENCE_RETENTION == "dvc":
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
                host_step=True,
                promotion_root=str(RUN_ROOT / "evidence"),
            )
        shell:
            "{params.command} > {log:q} 2>&1"

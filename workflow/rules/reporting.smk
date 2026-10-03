"""CPU report generation after the scientific stages succeed."""


if REPORT_ENABLED:
    rule build_run_report:
        input:
            training=str(TRAINING_MANIFEST),
            provenance=str(TRAINING_EVIDENCE_TERMINAL),
            dynamics_manifest=str(TRAINING_DYNAMICS_MANIFEST),
            dynamics=str(TRAINING_DYNAMICS),
            evaluation=str(EVALUATION_MANIFEST),
            evaluation_artifacts=EVALUATION_ARTIFACTS,
            dependencies=str(DEPENDENCY_CONTRACTS["reporting"]),
        output:
            manifest=str(REPORT_MANIFEST),
            html=str(REPORT_DIR / "report.html"),
            bundle=str(REPORT_DIR / "bundle_manifest.json"),
        threads:
            1
        log:
            str(RUN_ROOT / "logs/build_run_report.log")
        params:
            command=container_command(
                [
                    "python", "-m", "scripts.analysis.build_run_report",
                    "--run-root", CONTAINER_RUN,
                    "--output-dir", CONTAINER_RUN / REPORT_DIR.relative_to(RUN_ROOT),
                ],
                run_root=RUN_ROOT,
                source_root=SOURCE_ROOT,
                sif=SIF,
                dvc_remote=DVC_REMOTE,
                dvc_cache=DVC_CACHE,
                cpus=1,
                host_step=True,
            )
        shell:
            "{params.command} > {log:q} 2>&1"


    rule train_evaluate_report:
        input:
            str(REPORT_MANIFEST)

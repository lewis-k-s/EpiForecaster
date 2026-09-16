"""Run preparation, dependency contracts, and the DVC data boundary."""


rule prepare_run:
    input:
        config=str(CONFIG_PATH),
        inventory=str(SOURCE_FILE_LIST),
        dispatch=str(DISPATCH_REQUEST),
        published=str(PUBLISHED),
        dvc=DVC_POINTERS,
    output:
        PREPARE_OUTPUTS
    log:
        str(RUN_ROOT / "logs/prepare_run.log")
    shell:
        "PYTHONPATH={SOURCE_ROOT:q} {HOST_PYTHON:q} "
        "-m scripts.workflow.run_support prepare "
        "--config {input.config:q} --source-root {SOURCE_ROOT:q} "
        "--run-root {RUN_ROOT:q} > {log:q} 2>&1"


rule dependency_contract:
    input:
        prepared=str(PREPARED),
        config=str(CONFIG_PATH),
        source=lambda wildcards: STAGE_SOURCES[wildcards.stage],
    output:
        str(RUN_ROOT / ".state/dependencies/{stage}.json")
    params:
        command=lambda wildcards: dependency_command(wildcards.stage),
        source_fingerprint=lambda wildcards: STAGE_FINGERPRINTS[wildcards.stage],
    log:
        str(RUN_ROOT / "logs/dependencies-{stage}.log")
    wildcard_constraints:
        stage="|".join(ACTIVE_STAGES)
    shell:
        "{params.command} > {log:q} 2>&1"


rule stage_data:
    input:
        prepared=str(PREPARED),
        dependencies=str(DEPENDENCY_CONTRACTS["data"]),
        dvc=DVC_POINTERS,
    output:
        manifest=str(DATA_MANIFEST),
        marker=str(DATA_STAGED),
    threads:
        1
    log:
        str(RUN_ROOT / "logs/stage_data.log")
    params:
        command=container_command(
            [
                "python",
                "-m",
                "scripts.workflow.run_support",
                "stage-data",
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
            dvc_mode="pull",
            cpus=1,
            host_step=True,
        )
    shell:
        "{params.command} > {log:q} 2>&1"


rule complete_workflow:
    input:
        terminal=str(DEFAULT_TERMINAL)
    output:
        str(WORKFLOW_COMPLETE)
    log:
        str(RUN_ROOT / "logs/complete_workflow.log")
    shell:
        "PYTHONPATH={SOURCE_ROOT:q} {HOST_PYTHON:q} "
        "-m scripts.workflow.run_support complete "
        "--config {CONFIG_PATH:q} --run-root {RUN_ROOT:q} "
        "--terminal-artifact {input.terminal:q} > {log:q} 2>&1"

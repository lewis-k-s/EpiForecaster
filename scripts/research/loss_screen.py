"""Render the declared loss screen into immutable-workflow input templates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import yaml

from models.configs import EpiForecasterConfig
from scripts.workflow.run_config import validate_workflow_contract


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path("configs/research/loss_protocol_v1.yaml")


def screen_templates(protocol: dict, dataset_pointer: str) -> list[tuple[str, dict]]:
    """Return matched science identities without submitting or starting work."""
    pointer = Path(dataset_pointer)
    if pointer.is_absolute() or ".." in pointer.parts or pointer.suffix != ".dvc":
        raise ValueError("The dataset pointer must be source-relative")
    payload = yaml.safe_load((REPO_ROOT / pointer).read_text())["outs"][0]
    dataset = pointer.parent / payload["path"]
    expected = protocol["data"]["research_dataset_dvc_md5"]
    if expected is None or payload["md5"] != expected:
        raise ValueError(
            "The replacement dataset must pass QA and be pinned in the protocol"
        )
    if dataset.is_absolute() or ".." in dataset.parts:
        raise ValueError("The research dataset must be source-relative")
    screen = protocol["screen"]
    anchor = yaml.safe_load((REPO_ROOT / screen["anchor_campaign"]).read_text())
    split = anchor["cross_validation"]
    embedding = anchor["fixed_overrides"]["data"]["region2vec_path"]
    common = [
        f"data.dataset_path={dataset.as_posix()}",
        f"data.region2vec_path={embedding}",
        "training.split_strategy=research",
        f"training.research_role_manifest_path={protocol['data']['role_manifest']}",
        "training.research_evaluation_role=calibration",
        "training.research_confirmation_enabled=false",
        "training.node_split_strategy=crossval",
        "training.crossval_enabled=true",
        f"training.crossval_manifest_path={split['split_manifest']}",
        f"training.crossval_num_folds={len(split['fold_indices'])}",
        "training.crossval_holdout_role=test",
        f"training.selection_metric={protocol['selection']['metric']}",
        "training.retain_best_crps_checkpoint=true",
        f"training.epochs={screen['max_epochs']}",
        f"training.early_stopping_patience={screen['early_stopping_patience']}",
        "training.run_final_evaluation=false",
        "training.profiler.enabled=false",
        "training.plot_forecasts=false",
        "output.save_checkpoints=true",
    ]
    for key, value in anchor["fixed_overrides"]["model"]["observation_heads"].items():
        common.append(f"model.observation_heads.{key}={str(value).lower()}")
    templates = []
    # Paired settings are adjacent, so limited budgets do not spend all capacity
    # on one distribution before obtaining its matched control.
    for fold, seed, mobility, family, objective in itertools.product(
        screen["folds"],
        screen["paired_seeds"],
        screen["mobility"],
        screen["families"],
        screen["objectives"],
    ):
        identity = f"loss-b-{family}-{objective}-m{int(mobility)}-f{fold}-s{seed}"
        overrides = common + [
            f"training.crossval_fold_index={fold}",
            f"training.seed={seed}",
            f"model.type.mobility={str(mobility).lower()}",
            f"training.loss.joint.distribution_family={family}",
            f"training.loss.joint.distribution_df={screen['student_t_df']}",
            f"training.loss.joint.observation_loss={objective}",
        ]
        # Parse every generated config through the same canonical dataclasses.
        EpiForecasterConfig.load(REPO_ROOT / anchor["base_config"], overrides=overrides)
        template = {
            "run": {},
            "source": {},
            "container": {},
            "workflow": {"kind": "train", "target": "train_evaluate"},
            "dvc": {
                "targets": [
                    "data/files/mobility.zarr.dvc",
                    str(pointer),
                    f"{embedding}.dvc",
                ]
            },
            "training": {
                "base_config": anchor["base_config"],
                "experiment_name": protocol["protocol_id"],
                "model_id": identity,
                "max_batches": None,
                "overrides": overrides,
            },
            "evaluation": {
                "split": "val",
                "granular": True,
                "research_outputs": True,
                "overrides": [
                    "training.num_forecast_samples=1",
                    "training.plot_forecasts=false",
                ],
            },
            "resources": {
                "allocation": {
                    "account": "bsc08",
                    "partition": "acc",
                    "qos": "acc_debug",
                    "nodes": 1,
                    "tasks": 1,
                    "cpus_per_task": 20,
                    "gres": "gpu:1",
                    "runtime": protocol["execution"]["max_allocation_minutes"],
                }
            },
        }
        validate_workflow_contract(template)
        templates.append((identity, template))
    if len({identity for identity, _ in templates}) != len(templates):
        raise ValueError("Scientific identities must be unique")
    if len(templates) > protocol["budget"]["max_scientific_training_identities"]:
        raise ValueError("Screen exceeds the declared identity budget")
    return templates


def acceptance_templates(
    protocol: dict, dataset_pointer: str
) -> list[tuple[str, dict]]:
    """Use the first declared fold/seed for bounded pipeline acceptance only."""
    bounded = copy.deepcopy(protocol)
    bounded["screen"]["folds"] = protocol["screen"]["folds"][:1]
    bounded["screen"]["paired_seeds"] = protocol["screen"]["paired_seeds"][:1]
    acceptance = protocol["acceptance"]
    bounded["screen"]["max_epochs"] = acceptance["max_epochs"]
    bounded["screen"]["early_stopping_patience"] = acceptance["early_stopping_patience"]
    bounded["execution"]["max_allocation_minutes"] = acceptance[
        "max_allocation_minutes"
    ]
    templates = []
    for science_id, template in screen_templates(bounded, dataset_pointer):
        identity = science_id.replace("loss-b-", "loss-acceptance-", 1)
        training = template["training"]
        training["model_id"] = identity
        training["experiment_name"] += "-acceptance"
        training["max_batches"] = acceptance["max_batches_per_epoch"]
        training["overrides"].extend(
            [
                f"training.compile={str(acceptance['compile']).lower()}",
                f"training.compile_backward={str(acceptance['compile']).lower()}",
            ]
        )
        EpiForecasterConfig.load(
            REPO_ROOT / training["base_config"], overrides=training["overrides"]
        )
        templates.append((identity, template))
    return templates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-pointer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["screen", "acceptance"], default="screen")
    args = parser.parse_args()
    protocol_bytes = (REPO_ROOT / PROTOCOL).read_bytes()
    protocol = yaml.safe_load(protocol_bytes)
    render = acceptance_templates if args.stage == "acceptance" else screen_templates
    templates = render(protocol, args.dataset_pointer)
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for identity, template in templates:
        path = args.output / f"{identity}.yaml"
        path.write_text(yaml.safe_dump(template, sort_keys=False))
        records.append(
            {
                "science_id": identity,
                "template": path.name,
                "template_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (args.output / f"{args.stage}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
                "status": "prepared_not_submitted",
                "stage": args.stage,
                "scientific_selection_eligible": args.stage == "screen",
                "candidates": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

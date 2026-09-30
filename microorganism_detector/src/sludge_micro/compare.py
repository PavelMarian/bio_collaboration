"""Comparison of runs evaluated on one validation split by one evaluator.

The control is ``protocol.control_run`` or, when it is not set, the only run
whose architecture equals the configuration's model; ``protocol.control_dir``
takes the control from the experiments folder of an earlier stage. A candidate is accepted
when it passes the mandatory checks and its exact validation macro F1 is greater
than the control's; the selected run is the accepted candidate with the highest
macro F1, otherwise the control. Test results are not read. The comparison
records the code each run executed and checks that every run used the locked
split, the filtered train set and the control's validation, and that a run with
frozen BatchNorm kept its stored statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sludge_micro.config import ProjectConfig, load_config
from sludge_micro.messages import ProjectError
from sludge_micro.reporting import read_json, relative, write_json
from sludge_micro.runs import run_dirs, split_ids
from sludge_micro.split import lock_path


def executed_code(report_dir: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Code recorded when the run started, or the separate record of an earlier run."""
    record = manifest.get("code_at_start")
    if record is None and (report_dir / "code_at_start.json").is_file():
        record = read_json(report_dir / "code_at_start.json")
    if record is None:
        return None
    return {"commit": record["commit"], "package_sha256": record["package_sha256"]}


def train_images(manifest: dict[str, Any]) -> int | None:
    """Number of training images recorded by a torchvision or a YOLO run."""
    if "train_images" in manifest:
        return manifest["train_images"]
    dataset = manifest.get("yolo_dataset")
    return dataset["images"]["train"] if dataset else None


def exported_train_ids(config: ProjectConfig, run_id: str) -> set[str] | None:
    """Image ids of the exported YOLO train folder, if the run has one."""
    folder = run_dirs(config, run_id)[1] / "yolo_dataset" / "images" / "train"
    return {p.stem for p in folder.iterdir()} if folder.is_dir() else None


def configured_trainable(snapshot: ProjectConfig) -> dict[str, Any] | None:
    """Trained parameters that a torchvision run's configuration defines, without weights."""
    if snapshot.model.backend != "torchvision":
        return None
    from sludge_micro.model import build_detector, freeze_backbone
    from sludge_micro.train import trainable_parameters

    model = build_detector(snapshot)
    freeze_backbone(model, snapshot.model.trainable_backbone_layers)
    return trainable_parameters(model)


def metrics_files(config: ProjectConfig) -> list[Path]:
    """Validation metrics to compare: runs of the experiments folder and an earlier control.

    Raises:
        ProjectError: If ``protocol.control_dir`` names a control without metrics.
    """
    paths = sorted(config.resolve(config.paths.experiments_dir).glob("*/metrics.json"))
    protocol = config.protocol
    if protocol.control_dir is None:
        return paths
    control = config.resolve(protocol.control_dir) / str(protocol.control_run) / "metrics.json"
    if not control.is_file():
        raise ProjectError(
            "run_missing", run_id=protocol.control_run, path=relative(control.parent, config.root)
        )
    return [control, *(p for p in paths if p.parent.name != protocol.control_run)]


def run_records(config: ProjectConfig, expected: set[str]) -> list[dict[str, Any]]:
    """Validation metrics, data, code and settings of every compared run."""
    records = []
    for metrics_path in metrics_files(config):
        snapshot = load_config(metrics_path.parent / "config.yaml")
        metrics = read_json(metrics_path)
        manifest = read_json(metrics_path.parent / "manifest.json")
        exported = exported_train_ids(config, metrics["run_id"])
        records.append(
            {
                "run_id": metrics["run_id"],
                "report_dir": relative(metrics_path.parent, config.root),
                "architecture": snapshot.model.architecture,
                "split": metrics["split"],
                "confidence": metrics["confidence"],
                "macro_f1": metrics["macro_f1"],
                "macro_f1_classes": metrics["macro_f1_classes"],
                "images": metrics["images"],
                "objects": metrics["objects"],
                "ignore_regions": metrics["ignore_regions"],
                "classes": metrics["classes"],
                "split_manifest_sha256": (manifest["split_lock"] or {}).get("manifest_sha256"),
                "train_images": train_images(manifest),
                "train_ids_match": None if exported is None else exported == expected,
                "code": executed_code(metrics_path.parent, manifest),
                "train": {
                    "epochs": snapshot.train.epochs,
                    "lr": snapshot.train.lr,
                    "exclude_ignore_images": snapshot.train.exclude_ignore_images,
                    "batchnorm": snapshot.train.batchnorm,
                    "sampling": snapshot.train.sampling,
                    "augment": snapshot.train.augment.model_dump(mode="json"),
                },
                "batchnorm_statistics_changed": (manifest.get("batchnorm") or {}).get("changed"),
                "trainable_parameters": manifest.get("trainable_parameters"),
                "trainable_parameters_by_config": configured_trainable(snapshot),
            }
        )
    return records


def failed_checks(
    record: dict[str, Any], control: dict[str, Any], lock: str, expected: int
) -> list[str]:
    """Mandatory checks a run fails: split, train set, validation and frozen statistics."""
    checks = {
        "split": record["split_manifest_sha256"] == lock,
        "train": record["train_images"] == expected and record["train_ids_match"] is not False,
        "validation": all(
            record[k] == control[k] for k in ("split", "images", "objects", "ignore_regions")
        ),
        "batchnorm_statistics": record["train"]["batchnorm"] != "frozen"
        or record["batchnorm_statistics_changed"] == 0,
    }
    return [name for name, passed in checks.items() if not passed]


def control_candidates(config: ProjectConfig, records: list[dict[str, Any]]) -> list[dict]:
    """Runs that match the configured control: its run id or, if unset, its architecture."""
    if config.protocol.control_run is not None:
        return [r for r in records if r["run_id"] == config.protocol.control_run]
    return [r for r in records if r["architecture"] == config.model.architecture]


def decide(
    config: ProjectConfig, records: list[dict[str, Any]], lock: str, expected: int
) -> dict[str, Any]:
    """Apply the pre-registered acceptance rule to the validation results.

    Args:
        lock: SHA-256 of the locked split manifest.
        expected: Number of training images after the configured filter.
    """
    controls = control_candidates(config, records)
    if len(controls) != 1:
        return {"control": None, "accepted": [], "selected": None, "reason": "control run missing"}
    control = controls[0]
    failed = {r["run_id"]: failed_checks(r, control, lock, expected) for r in records}
    accepted = [
        r["run_id"]
        for r in records
        if r is not control
        and not failed[r["run_id"]]
        and r["macro_f1"] is not None
        and r["macro_f1"] > control["macro_f1"]
    ]
    best = max(
        (r for r in records if r["run_id"] in accepted), key=lambda r: r["macro_f1"], default=None
    )
    return {
        "control": control["run_id"],
        "control_macro_f1": control["macro_f1"],
        "failed_checks": {k: v for k, v in failed.items() if v},
        "accepted": accepted,
        "selected": best["run_id"] if best else control["run_id"],
        "rule": config.protocol.acceptance,
    }


def run_comparison(config: ProjectConfig) -> dict[str, Any]:
    """Write ``comparison.json`` for the runs of the configured experiments folder."""
    expected = split_ids(config, "train")
    records = run_records(config, expected)
    lock = read_json(lock_path(config))["manifest_sha256"]
    decision = decide(config, records, lock, len(expected))
    if decision["control"] is None:
        raise ProjectError("compare_control_missing", run_id=config.protocol.control_run)
    document = {
        "runs": records,
        "decision": decision,
        "split_manifest_sha256": lock,
        "train_images_expected": len(expected),
        "same_split": all(r["split_manifest_sha256"] == lock for r in records),
        "same_train": all(
            r["train_images"] == len(expected) and r["train_ids_match"] is not False
            for r in records
        ),
        "same_validation": len({(r["images"], r["objects"], r["ignore_regions"]) for r in records})
        == 1,
    }
    write_json(config.resolve(config.paths.experiments_dir) / "comparison.json", document)
    return document

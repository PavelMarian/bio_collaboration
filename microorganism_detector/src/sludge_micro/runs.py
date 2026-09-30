"""Run directories, evaluation of a run on a split and reproducibility manifests.

Validation selects the confidence threshold. Test is evaluated once for a
frozen run: before predicting, the checkpoint, the configuration and its
snapshot are checked against the run manifest. The fact, the checks and the
executed code are recorded in ``test_usage.json`` of the experiments folder and
a second test evaluation is refused. The test report also records count errors
of the same detections. Reports reuse saved predictions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from sludge_micro.archives import sha256_file
from sludge_micro.config import ProjectConfig, load_config
from sludge_micro.dataset import Sample, build_samples
from sludge_micro.evaluate import count_errors, match_all, select_confidence, summarise
from sludge_micro.imaging import read_bgr
from sludge_micro.infer import PREDICTION_COLUMNS, Predictor, raw_frame
from sludge_micro.messages import ProjectError
from sludge_micro.model import load_trained, torch_device
from sludge_micro.prepare import load_prepared
from sludge_micro.reporting import read_json, relative, write_json
from sludge_micro.runtime import code_at_start, code_version, environment
from sludge_micro.split import load_split, lock_path


def run_dirs(config: ProjectConfig, run_id: str) -> tuple[Path, Path]:
    """Return the report directory and the artifact directory of a run."""
    return (
        config.resolve(config.paths.experiments_dir) / run_id,
        config.resolve(config.paths.outputs_dir) / run_id,
    )


def checkpoint_of(config: ProjectConfig, run_id: str) -> Path:
    """Return the checkpoint of a run or fail when it is absent."""
    path = run_dirs(config, run_id)[1] / "model.pt"
    if not path.is_file():
        raise ProjectError("run_missing", run_id=run_id, path=relative(path.parent, config.root))
    return path


def run_predictor(config: ProjectConfig, run_id: str, confidence: float) -> Predictor:
    """Load the predictor of a run for its configured backend."""
    path = checkpoint_of(config, run_id)
    if config.model.backend == "ultralytics":
        from sludge_micro.yolo import UltralyticsPredictor

        return UltralyticsPredictor(path, config, confidence)
    model, _ = load_trained(path, config)
    return Predictor(model, config.classes.names, confidence, torch_device(config.runtime.device))


def split_ids(config: ProjectConfig, split: str) -> set[str]:
    """Image ids of one locked split after the configured training filter."""
    manifest = load_split(config)
    ids = set(manifest.loc[manifest["split"] == split, "image_id"])
    if split == "train" and config.train.exclude_ignore_images:
        _, anns = load_prepared(config)
        ids -= set(anns.loc[anns["supervision"] == "ignore", "image_id"])
    return ids


def split_samples(config: ProjectConfig, split: str) -> tuple[list[Sample], pd.DataFrame]:
    """Return samples and supervision rows of one locked split."""
    ids = split_ids(config, split)
    images, anns = load_prepared(config)
    images, anns = images[images["image_id"].isin(ids)], anns[anns["image_id"].isin(ids)]
    return build_samples(config, images, anns), anns


def capture_period(part: pd.DataFrame) -> list[str] | None:
    """First and last confirmed capture time, or ``None`` when any time is unknown."""
    times = part["captured_at"]
    if part.empty or times.isna().any():
        return None
    return [min(times), max(times)]


def references(anns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split supervision rows into target references and ignore regions."""
    cols = ["image_id", "class_name", "x1", "y1", "x2", "y2"]
    target = anns[anns["supervision"] == "target"][cols].reset_index(drop=True)
    ignore = anns[anns["supervision"] == "ignore"][cols[:1] + cols[2:]].reset_index(drop=True)
    return target, ignore


def predict_samples(predictor: Predictor, samples: list[Sample]) -> pd.DataFrame:
    """Return raw predictions above the detector minimum score for every sample."""
    frames = [
        raw_frame(s.image_id, predictor.raw(read_bgr(s.path)), predictor.classes) for s in samples
    ]
    table = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PREDICTION_COLUMNS)
    )
    return table.sort_values(["image_id", "score"], ascending=[True, False], kind="stable")


def metrics_record(
    config: ProjectConfig,
    split: str,
    preds: pd.DataFrame,
    anns: pd.DataFrame,
    confidence: float | None,
) -> dict[str, Any]:
    """Evaluate predictions; select the threshold when none is given."""
    target, ignore = references(anns)
    image_ids = sorted(set(anns["image_id"]) | set(preds["image_id"]))
    det, ref = match_all(preds, target, ignore, config.evaluation)
    curve = None
    if confidence is None:
        confidence, curve = select_confidence(det, ref, config.classes.names, config.evaluation)
    result = summarise(det, ref, config.classes.names, confidence)
    if split == "test":
        result["counting"] = count_errors(det, ref, image_ids, config.classes.names, confidence)
    manifest = load_split(config)
    part = manifest[manifest["split"] == split]
    return result | {
        "split": split,
        "images": len(image_ids),
        "objects": len(target),
        "ignore_regions": len(ignore),
        "selection_curve": curve,
        "confidence_source": (
            "validation_selection" if curve is not None else "frozen_from_validation"
        ),
        "period": capture_period(part),
        "evaluation": config.evaluation.model_dump(mode="json"),
        "postprocess": config.postprocess.model_dump(mode="json"),
    }


def test_usage_path(config: ProjectConfig) -> Path:
    """Return the record of the single test evaluation."""
    return config.resolve(config.paths.experiments_dir) / "test_usage.json"


def run_checks(config: ProjectConfig, run_id: str) -> dict[str, Any]:
    """Compare the checkpoint, configuration and split of a run with its manifest."""
    report_dir, _ = run_dirs(config, run_id)
    manifest = read_json(report_dir / "manifest.json")
    snapshot = report_dir / "config.yaml"
    checkpoint = sha256_file(checkpoint_of(config, run_id))
    lock = read_json(lock_path(config))["manifest_sha256"]
    checks = {
        "checkpoint_matches_manifest": checkpoint == manifest["inputs"]["checkpoint"]["sha256"],
        "snapshot_matches_manifest": sha256_file(snapshot) == manifest["config_sha256"],
        "config_equals_snapshot": load_config(snapshot).model_dump(mode="json", exclude={"root"})
        == config.model_dump(mode="json", exclude={"root"}),
        "split_matches_lock": (manifest.get("split_lock") or {}).get("manifest_sha256") == lock,
    }
    return {
        "checkpoint_sha256": checkpoint,
        "config_sha256": manifest["config_sha256"],
        "split_manifest_sha256": lock,
    } | checks


def failed_run_checks(checks: dict[str, Any]) -> list[str]:
    """Names of the failed checks of ``run_checks``."""
    return sorted(name for name, ok in checks.items() if ok is False)


def frozen_run_check(config: ProjectConfig, run_id: str) -> dict[str, Any]:
    """Check that the checkpoint and configuration are the ones the run manifest records.

    Raises:
        ProjectError: If the checkpoint, the configuration snapshot, the configuration or
            the split differ from the manifest of the run.
    """
    checks = run_checks(config, run_id)
    failed = failed_run_checks(checks)
    if failed:
        raise ProjectError("test_run_mismatch", run_id=run_id, checks=", ".join(failed))
    return checks


def test_preconditions(config: ProjectConfig, run_id: str) -> tuple[float, dict[str, Any]]:
    """Refuse a second test evaluation; return the validation threshold and the checks.

    The checks hold the comparison of the run with its manifest and the code that
    runs the evaluation, captured before anything is predicted.
    """
    if test_usage_path(config).is_file():
        raise ProjectError("test_already_used", path=relative(test_usage_path(config), config.root))
    validation = run_dirs(config, run_id)[0] / "metrics.json"
    if not validation.is_file():
        raise ProjectError("confidence_unselected", run_id=run_id)
    checks = {"run_check": frozen_run_check(config, run_id), "code": code_at_start(config.root)}
    return read_json(validation)["confidence"], checks


def usage_record(
    config: ProjectConfig, run_id: str, record: dict[str, Any], checks: dict[str, Any]
) -> dict[str, Any]:
    """Record of the single test evaluation: run, threshold, files, checks and code."""
    report_dir, out_dir = run_dirs(config, run_id)
    predictions = out_dir / "predictions_test.parquet"
    return {
        "run_id": run_id,
        "confidence": record["confidence"],
        "images": record["images"],
        "objects": record["objects"],
        "metrics": relative(report_dir / "test_metrics.json", config.root),
        "predictions": relative(predictions, config.root),
        "predictions_sha256": sha256_file(predictions),
    } | checks


def evaluate_run(config: ProjectConfig, run_id: str, split: str) -> dict[str, Any]:
    """Evaluate a run on validation (threshold selection) or once on test."""
    report_dir, out_dir = run_dirs(config, run_id)
    confidence, checks = test_preconditions(config, run_id) if split == "test" else (None, {})
    predictor = run_predictor(config, run_id, 0.0)
    samples, anns = split_samples(config, split)
    preds = predict_samples(predictor, samples)
    preds.to_parquet(out_dir / f"predictions_{split}.parquet", index=False)
    record = metrics_record(config, split, preds, anns, confidence) | {"run_id": run_id}
    name = "metrics.json" if split == "validation" else "test_metrics.json"
    write_json(report_dir / name, record)
    if split == "test":
        write_json(test_usage_path(config), usage_record(config, run_id, record, checks))
    return record


def file_digest(path: Path | None, root: Path) -> dict[str, Any] | None:
    """Return the relative path and SHA-256 of an input file, if present."""
    if path is None or not path.is_file():
        return None
    return {"path": relative(path, root), "sha256": sha256_file(path)}


def write_run_manifest(config: ProjectConfig, run_id: str, extra: dict[str, Any]) -> dict[str, Any]:
    """Record code, configuration, inputs, environment and class map of a run."""
    report_dir, out_dir = run_dirs(config, run_id)
    config_text = yaml.safe_dump(
        config.model_dump(mode="json", exclude={"root"}), allow_unicode=True
    )
    (report_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    weights = (
        config.resolve(config.paths.pretrained_weights) if config.paths.pretrained_weights else None
    )
    manifest = {
        "run_id": run_id,
        "code": code_version(config.root),
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "class_map": [c.model_dump(mode="json") for c in config.classes.target],
        "split_lock": read_json(lock_path(config)) if lock_path(config).is_file() else None,
        "inputs": {
            "pretrained_weights": file_digest(weights, config.root),
            "checkpoint": file_digest(out_dir / "model.pt", config.root),
        },
        "environment": environment(config.runtime.device),
        "seed": config.runtime.seed,
    } | extra
    write_json(report_dir / "manifest.json", manifest)
    return manifest


def rebuild_test_report(
    config: ProjectConfig, scratch: ProjectConfig, run_id: str
) -> dict[str, Any]:
    """Recompute the test report from saved predictions without predicting again.

    References come from the locked split of ``config``; the report is written to
    the experiments folder of ``scratch``.
    """
    report_dir, out_dir = run_dirs(config, run_id)
    preds = pd.read_parquet(out_dir / "predictions_test.parquet")
    confidence = read_json(report_dir / "metrics.json")["confidence"]
    _, anns = split_samples(config, "test")
    record = metrics_record(config, "test", preds, anns, confidence) | {"run_id": run_id}
    write_json(run_dirs(scratch, run_id)[0] / "test_metrics.json", record)
    return record

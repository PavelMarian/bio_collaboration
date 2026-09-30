"""Check of the hand-over of a detector to the Pavel pipeline on real images.

A trained run is exported with ``export_for_pavel`` and checked on its
validation frames (``run_handover``); a delivered model is exported with
``export_passport`` and checked on a folder of images (``run_passport_check``).
A temporary copy of the pipeline package ``cv_module`` runs ``AnalysisPipeline``
in a separate process with an empty segmentation stub and the pipeline's own
detector backend. Its detections and counts are compared with the project's own
inference on the same images. The pipeline folder itself is never modified;
architectures that need a patched pipeline are refused.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from sludge_micro.adapter import export_for_pavel, export_passport
from sludge_micro.archives import sha256_file
from sludge_micro.config import ProjectConfig
from sludge_micro.infer import Predictor, discover_images, predict_images
from sludge_micro.messages import ProjectError
from sludge_micro.passport import ModelPassport, passport_predictor
from sludge_micro.prepare import load_prepared
from sludge_micro.reporting import read_json, write_json
from sludge_micro.runs import checkpoint_of, run_dirs, run_predictor, split_ids

PIPELINE_SCRIPT = """
import json, sys
from cv_module.config import load_config
from cv_module.inference.pipeline import AnalysisPipeline
cfg = load_config(sys.argv[1])
if cfg.microorganism_backend == "torchvision":
    from cv_module.backends.torchvision import TorchvisionCocoModel
    model = TorchvisionCocoModel(cfg.microorganism_model, cfg.microorganism_coco_annotations,
                                 cfg.microorganism_architecture, cfg.device)
    detect = lambda im: model.detect(im, confidence=cfg.microorganism_confidence,
                                     max_detections=cfg.max_detections)
else:
    from cv_module.backends.ultralytics import UltralyticsModel
    model = UltralyticsModel(cfg.microorganism_model)
    detect = lambda im: model.detect(
        im, image_size=cfg.image_size, confidence=cfg.microorganism_confidence,
        iou=cfg.iou_threshold, device=cfg.device, max_detections=cfg.max_detections,
    )
class Backend:
    def segment(self, image):
        return []
    def detect(self, image):
        return detect(image)
    def model_classes(self):
        return {cfg.floc_class_name}, set(model.classes)
pipeline = AnalysisPipeline(cfg, Backend())
errors = pipeline.validate_model_classes()
summary = pipeline.run(sys.argv[2], sys.argv[3])
print(json.dumps({"errors": errors, "summary": summary}))
"""


def pipeline_copy(pipeline: Path | None, folder: Path, architecture: str) -> Path:
    """Copy the pipeline package ``cv_module`` into a working folder.

    Raises:
        ProjectError: If the architecture needs a patched pipeline, or the pipeline
            folder is not set or holds no ``cv_module``.
    """
    if architecture == "retinanet_resnet50_fpn_v2":
        raise ProjectError("pavel_patch_required", architecture=architecture)
    if pipeline is None or not (pipeline / "cv_module").is_dir():
        raise ProjectError("pavel_pipeline_missing", path=pipeline)
    copy = folder / "pavel-pipeline"
    ignore = shutil.ignore_patterns("__pycache__")
    shutil.copytree(pipeline / "cv_module", copy / "cv_module", ignore=ignore)
    return copy


def pipeline_config(export: Path, device: str) -> Path:
    """Complete the exported block with a floc entry that the stub never loads."""
    block = yaml.safe_load((export / "microorganism_block.yaml").read_text(encoding="utf-8"))
    (export / "models" / "floc_stub.pt").write_bytes(b"segmentation stub, not loaded")
    block["models"]["floc_segmentation"] = {
        "backend": "ultralytics",
        "weights": "models/floc_stub.pt",
    }
    block["inference"]["device"] = device
    path = export / "pipeline.yaml"
    path.write_text(yaml.safe_dump(block, allow_unicode=True), encoding="utf-8")
    return path


def frames(config: ProjectConfig, folder: Path, limit: int) -> Path:
    """Copy the first validation frames under their image ids."""
    images, _ = load_prepared(config)
    ids = sorted(split_ids(config, "validation"))[:limit]
    target = folder / "input"
    target.mkdir()
    for row in images[images["image_id"].isin(ids)].itertuples():
        source = config.resolve(row.path)
        shutil.copyfile(source, target / f"{row.image_id}{source.suffix.lower()}")
    return target


def run_pipeline(copy: Path, pipeline_yaml: Path, source: Path, out: Path) -> dict[str, Any]:
    """Run the pipeline script inside the copy and return its summary."""
    env = os.environ | {"YOLO_OFFLINE": "true"}
    result = subprocess.run(
        [sys.executable, "-c", PIPELINE_SCRIPT, str(pipeline_yaml), str(source), str(out)],
        cwd=copy,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def compare_outputs(
    predictor: Predictor, suffixes: Sequence[str], source: Path, out: Path
) -> dict[str, Any]:
    """Compare pipeline detections and counts with the project's own inference."""
    ours = {p.image_id: p for p in predict_images(predictor, discover_images(source, suffixes))}
    theirs = json.loads((out / "microorganism_detections.json").read_text(encoding="utf-8"))
    with (out / "microorganism_counts.csv").open(encoding="utf-8-sig") as handle:
        counts = {row["image_id"]: row for row in csv.DictReader(handle)}
    same = counts_ok = 0
    for doc in theirs:
        mine = ours[doc["image_id"]].objects
        pairs = list(zip(doc["objects"], mine, strict=False))
        same += len(doc["objects"]) == len(mine) and all(
            t["class"] == o.class_name
            and max(abs(a - b) for a, b in zip(t["bbox_xyxy"], o.bbox_xyxy, strict=True)) <= 0.01
            and abs(t["confidence"] - o.confidence) <= 1e-5
            for t, o in pairs
        )
        tally = Counter(t["class"] for t in doc["objects"])
        counts_ok += all(
            int(counts[doc["image_id"]][c]) == tally.get(c, 0) for c in predictor.classes
        )
    return {
        "images": len(theirs),
        "detections_match": same,
        "counts_match_detections": counts_ok,
        "objects": sum(len(d["objects"]) for d in theirs),
        "count_columns": list(next(iter(counts.values())).keys()) if counts else [],
    }


def check_with_pipeline(
    pipeline: Path | None, export: Path, source: Path, predictor: Predictor, settings: dict
) -> dict[str, Any]:
    """Run the pipeline with an export on a folder and compare it with own inference.

    Args:
        settings: ``architecture``, ``device`` and image ``suffixes``.
    """
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        copy = pipeline_copy(pipeline, work, settings["architecture"])
        out = work / "pipeline_output"
        yaml_path = pipeline_config(export, settings["device"])
        result = run_pipeline(copy, yaml_path, source, out)
        comparison = compare_outputs(predictor, settings["suffixes"], source, out)
    return {"class_errors": result["errors"]} | comparison


def run_handover(config: ProjectConfig, run_id: str, limit: int) -> dict[str, Any]:
    """Export a trained run and check the pipeline hand-over on its validation frames."""
    confidence = read_json(run_dirs(config, run_id)[0] / "metrics.json")["confidence"]
    export = run_dirs(config, run_id)[1] / "pavel_export"
    checksums = export_for_pavel(config, checkpoint_of(config, run_id), confidence, export)
    pipeline = config.resolve(config.paths.pavel_pipeline) if config.paths.pavel_pipeline else None
    settings = {
        "architecture": config.model.architecture,
        "device": config.runtime.device,
        "suffixes": config.audit.image_suffixes,
    }
    with tempfile.TemporaryDirectory() as folder:
        source = frames(config, Path(folder), limit)
        predictor = run_predictor(config, run_id, confidence)
        checked = check_with_pipeline(pipeline, export, source, predictor, settings)
    report = {
        "run_id": run_id,
        "architecture": config.model.architecture,
        "confidence": confidence,
        "export_sha256": checksums,
        "checkpoint_sha256": sha256_file(checkpoint_of(config, run_id)),
        "export_weights_identical": checksums["weights"]
        == sha256_file(checkpoint_of(config, run_id)),
    } | checked
    write_json(run_dirs(config, run_id)[0] / "pavel_handover.json", report)
    return report


def run_passport_check(
    passport: ModelPassport, weights: Path, pipeline: Path, source: Path, report: Path
) -> dict[str, Any]:
    """Check a delivered model with the pipeline on a folder of images and write a report."""
    predictor = passport_predictor(passport, weights)
    settings = {
        "architecture": passport.detector.architecture,
        "device": "cpu",
        "suffixes": passport.image.formats,
    }
    with tempfile.TemporaryDirectory() as folder:
        export = Path(folder) / "export"
        checksums = export_passport(passport, weights, export)
        checked = check_with_pipeline(pipeline, export, source, predictor, settings)
    record = {
        "run_id": passport.run_id,
        "confidence": passport.detector.confidence,
        "weights_sha256": passport.weights.sha256,
        "export_sha256": checksums,
        "export_weights_identical": checksums["weights"] == passport.weights.sha256,
        "input_folder": source.name,
    } | checked
    write_json(report, record)
    return record

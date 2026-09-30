"""Hand-over of detections to the Pavel pipeline.

Two paths are supported:

- in-process: ``PavelDetector.detect(image_bgr)`` returns a new list of the
  pipeline's frozen ``Detection`` records;
- file-based: ``export_for_pavel`` (a trained run) and ``export_passport`` (a
  delivered model checked against its passport) write the checkpoint, a COCO
  categories file whose ids equal the model labels, and the matching
  configuration block for ``TorchvisionCocoModel``, which maps label ``k`` to
  the ``k``-th category sorted by id.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sludge_micro.archives import sha256_file
from sludge_micro.config import ProjectConfig
from sludge_micro.infer import Predictor
from sludge_micro.messages import ProjectError
from sludge_micro.model import load_trained
from sludge_micro.passport import ModelPassport, passport_model


def pavel_detection_type() -> type:
    """Import the pipeline's ``Detection`` type.

    Raises:
        ProjectError: If the pipeline package is not installed.
    """
    try:
        from cv_module.inference.types import Detection
    except ImportError as exc:
        raise ProjectError("pavel_missing") from exc
    return Detection


class PavelDetector:
    """Expose the microorganism detector through the pipeline ``detect`` contract."""

    def __init__(self, predictor: Predictor) -> None:
        """Wrap a predictor whose confidence threshold is already fixed."""
        self.predictor = predictor
        self._detection = pavel_detection_type()

    @property
    def classes(self) -> set[str]:
        """Return the class names the detector can output."""
        return set(self.predictor.classes)

    def detect(self, image_bgr: np.ndarray) -> list[Any]:
        """Return a new list of pipeline ``Detection`` records for one BGR image."""
        return [
            self._detection(o.class_name, tuple(o.bbox_xyxy), float(o.confidence))
            for o in self.predictor.predict_bgr(image_bgr)
        ]


def categories_document(labels: Sequence[tuple[int, str]]) -> dict[str, Any]:
    """COCO document whose category ids equal the internal model labels."""
    return {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": label, "name": name, "supercategory": "microorganism"} for label, name in labels
        ],
    }


def pavel_block(
    architecture: str, confidence: float, max_detections: int, names: Sequence[str]
) -> dict[str, Any]:
    """Configuration block for the pipeline, with paths relative to the export folder."""
    return {
        "models": {
            "microorganism_detection": {
                "backend": "torchvision",
                "architecture": architecture,
                "weights": "models/microorganisms_fasterrcnn.pt",
                "coco_annotations": "annotations/microorganism_instances.json",
            }
        },
        "inference": {
            "microorganism_confidence": confidence,
            "max_detections": max_detections,
        },
        "microorganism_classes": list(names),
    }


def write_torchvision_export(
    checkpoint: Path, out_dir: Path, categories: dict[str, Any], block: dict[str, Any]
) -> dict[str, str]:
    """Copy torchvision weights and write the categories file and the pipeline block."""
    files = {
        "weights": out_dir / "models" / "microorganisms_fasterrcnn.pt",
        "categories": out_dir / "annotations" / "microorganism_instances.json",
        "config": out_dir / "microorganism_block.yaml",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, files["weights"])
    files["categories"].write_text(json.dumps(categories, indent=2), encoding="utf-8")
    files["config"].write_text(yaml.safe_dump(block, allow_unicode=True), encoding="utf-8")
    return {name: sha256_file(path) for name, path in files.items()}


def export_passport(passport: ModelPassport, weights: Path, out_dir: Path) -> dict[str, str]:
    """Write the pipeline files of a delivered model after checking it against its passport."""
    passport_model(passport, weights)
    d = passport.detector
    return write_torchvision_export(
        weights,
        out_dir,
        categories_document([(c.label, c.name) for c in passport.classes]),
        pavel_block(d.architecture, d.confidence, d.max_detections, passport.names),
    )


def yolo_block(config: ProjectConfig, confidence: float) -> dict[str, Any]:
    """Configuration block for the pipeline's Ultralytics backend."""
    return {
        "models": {
            "microorganism_detection": {
                "backend": "ultralytics",
                "weights": "models/microorganisms_yolo.pt",
            }
        },
        "inference": {
            "microorganism_confidence": confidence,
            "image_size": config.model.yolo.image_size,
            "iou_threshold": config.postprocess.nms_iou,
            "max_detections": config.postprocess.max_detections,
        },
        "microorganism_classes": list(config.classes.names),
    }


def export_yolo_for_pavel(
    config: ProjectConfig, checkpoint: Path, confidence: float, out_dir: Path
) -> dict[str, str]:
    """Copy YOLO weights with a checked class order and write the pipeline block."""
    from sludge_micro.yolo import UltralyticsPredictor

    UltralyticsPredictor(checkpoint, config, confidence)
    files = {
        "weights": out_dir / "models" / "microorganisms_yolo.pt",
        "config": out_dir / "microorganism_block.yaml",
    }
    files["weights"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, files["weights"])
    files["config"].write_text(
        yaml.safe_dump(yolo_block(config, confidence), allow_unicode=True), encoding="utf-8"
    )
    return {name: sha256_file(path) for name, path in files.items()}


def export_for_pavel(
    config: ProjectConfig, checkpoint: Path, confidence: float, out_dir: Path
) -> dict[str, str]:
    """Write the files the pipeline needs and return their SHA-256 sums."""
    if config.model.backend == "ultralytics":
        return export_yolo_for_pavel(config, checkpoint, confidence, out_dir)
    load_trained(checkpoint, config)
    return write_torchvision_export(
        checkpoint,
        out_dir,
        categories_document([(t.label, t.name) for t in config.classes.target]),
        pavel_block(
            config.model.architecture,
            confidence,
            config.postprocess.max_detections,
            config.classes.names,
        ),
    )

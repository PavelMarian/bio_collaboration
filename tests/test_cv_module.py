from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from cv_module.backends.torchvision import read_coco_categories
from cv_module.config import DEFAULT_CLASSES, ModuleConfig, load_config
from cv_module.inference.pipeline import AnalysisPipeline
from cv_module.inference.types import Detection, Segmentation
from cv_module.masks.features import extract_features


class FakeBackend:
    def segment(self, image_bgr):
        mask = np.zeros(image_bgr.shape[:2], dtype=bool)
        mask[2:6, 3:11] = True
        return [Segmentation(mask, 0.9, "floc")]

    def detect(self, image_bgr):
        return [Detection("rotifers", (1.0, 2.0, 7.0, 8.0), 0.8)]


def test_known_mask_features():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:6, 3:11] = True
    values = extract_features(image, mask)
    assert values["area_px"] == 32
    assert values["elongation"] == 2.0
    assert values["area_fraction"] == 32 / 200
    assert (values["mean_r"], values["mean_g"], values["mean_b"]) == (30, 20, 10)


def test_complete_pipeline_outputs(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    image_path = source / "sample.jpg"
    assert cv2.imwrite(str(image_path), np.full((10, 20, 3), 100, dtype=np.uint8))
    config = ModuleConfig(tmp_path / "floc.pt", tmp_path / "micro.pt", microorganism_classes=DEFAULT_CLASSES)
    summary = AnalysisPipeline(config, FakeBackend()).run(source, tmp_path / "output")
    assert summary["images_processed"] == 1
    assert summary["flocs_detected"] == 1
    with (tmp_path / "output" / "combined_analysis.csv").open(encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert row["floc_count"] == "1"
    assert row["rotifers"] == "1"
    assert row["nematoda"] == "0"
    detections = json.loads((tmp_path / "output" / "microorganism_detections.json").read_text())
    assert detections[0]["objects"][0]["class"] == "rotifers"


def test_coco_categories_are_mapped_by_id(tmp_path):
    path = tmp_path / "instances.json"
    path.write_text(json.dumps({"categories": [{"id": 7, "name": "rotifers"}, {"id": 2, "name": "nematoda"}]}))
    assert read_coco_categories(path) == ("nematoda", "rotifers")


def test_coco_backend_config_uses_yaml_relative_paths(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """models:
  floc_segmentation:
    backend: torchvision
    weights: models/floc.pt
    coco_annotations: annotations/floc.json
  microorganism_detection:
    backend: ultralytics
    weights: models/micro.pt
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.floc_backend == "torchvision"
    assert config.microorganism_backend == "ultralytics"
    assert config.floc_model == (tmp_path / "models" / "floc.pt").resolve()
    assert config.floc_coco_annotations == (tmp_path / "annotations" / "floc.json").resolve()


def test_coco_example_config_matches_detector_label_order():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config.coco.example.yaml")
    assert config.microorganism_backend == "torchvision"
    assert config.microorganism_architecture == "fasterrcnn_resnet50_fpn_v2"
    assert config.microorganism_model == (root / "models" / "microorganisms_fasterrcnn.pt").resolve()
    assert config.microorganism_coco_annotations == (root / "annotations" / "microorganism_instances.json").resolve()
    assert read_coco_categories(config.microorganism_coco_annotations) == DEFAULT_CLASSES
    assert config.microorganism_classes == DEFAULT_CLASSES
    assert config.microorganism_confidence == 0.20

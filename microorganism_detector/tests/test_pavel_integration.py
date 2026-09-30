"""Integration with the Pavel pipeline through its own interfaces.

Segmentation is a synthetic stub used only here; these tests check the
hand-over of detections and counts, not segmentation quality.
"""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest
import torch
from cv_module.backends.torchvision import TorchvisionCocoModel
from cv_module.config import ModuleConfig
from cv_module.inference.pipeline import AnalysisPipeline
from cv_module.inference.types import Detection, Segmentation

from sludge_micro.adapter import PavelDetector, export_for_pavel
from sludge_micro.infer import Predictor
from sludge_micro.model import build_detector, save_checkpoint
from sludge_micro.types import ObjectDetection


class StubPredictor:
    """Predictor stand-in with fixed detections."""

    classes = (
        "attached_ciliates",
        "filamentous_bacteria",
        "rotifers",
        "testate_amoebae",
        "free_swimming_ciliates",
        "nematoda",
        "gastrotrichs",
    )

    def predict_bgr(self, image_bgr):
        return (
            ObjectDetection("rotifers", (1.0, 2.0, 7.0, 8.0), 0.8),
            ObjectDetection("rotifers", (3.0, 3.0, 9.0, 9.0), 0.6),
        )


class Backend:
    """Pipeline backend: synthetic segmentation stub plus the real adapter."""

    def __init__(self, detector):
        self.detector = detector

    def segment(self, image_bgr):
        mask = np.zeros(image_bgr.shape[:2], dtype=bool)
        mask[2:6, 3:11] = True
        return [Segmentation(mask, 0.9, "floc")]

    def detect(self, image_bgr):
        return self.detector.detect(image_bgr)

    def model_classes(self):
        return {"floc"}, self.detector.classes


def test_adapter_returns_new_list_of_pipeline_records(helpers):
    detector = PavelDetector(StubPredictor())
    first = detector.detect(helpers.synthetic_image(1))
    assert all(isinstance(d, Detection) for d in first)
    assert first is not detector.detect(helpers.synthetic_image(1))
    with pytest.raises(AttributeError):
        first[0].confidence = 1.0


def test_pipeline_writes_detections_and_counts_from_adapter(tmp_path, helpers, config):
    source = tmp_path / "input"
    source.mkdir()
    cv2.imwrite(str(source / "sample.bmp"), helpers.synthetic_image(2, 20, 10))
    module = ModuleConfig(
        tmp_path / "f.pt", tmp_path / "m.pt", microorganism_classes=config.classes.names
    )
    pipeline = AnalysisPipeline(module, Backend(PavelDetector(StubPredictor())))
    assert pipeline.validate_model_classes() == []
    summary = pipeline.run(source, tmp_path / "out")
    assert summary["microorganisms_detected"] == 2
    document = json.loads((tmp_path / "out" / "microorganism_detections.json").read_text())
    assert [o["class"] for o in document[0]["objects"]] == ["rotifers", "rotifers"]
    with (tmp_path / "out" / "microorganism_counts.csv").open(encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert row["rotifers"] == "2" and row["nematoda"] == "0"
    assert set(row) == {"image_id", *config.classes.names}


@pytest.mark.slow
def test_exported_checkpoint_loads_in_pipeline_with_same_classes_and_outputs(
    tmp_path, helpers, config
):
    torch.manual_seed(0)
    model = build_detector(config)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(model, checkpoint, config, {"run_id": "synthetic"})
    export_for_pavel(config, checkpoint, 0.1, tmp_path / "export")
    upstream = TorchvisionCocoModel(
        tmp_path / "export" / "models" / "microorganisms_fasterrcnn.pt",
        tmp_path / "export" / "annotations" / "microorganism_instances.json",
        config.model.architecture,
        "cpu",
    )
    assert upstream.categories == config.classes.names
    image = helpers.synthetic_image(5, 96, 72)
    theirs = upstream.detect(
        image, confidence=0.1, max_detections=config.postprocess.max_detections
    )
    ours = Predictor(build_detector(config), config.classes.names, 0.1, torch.device("cpu"))
    ours.model.load_state_dict(torch.load(checkpoint, weights_only=False)["model"])
    mine = ours.predict_bgr(image)
    assert [d.class_name for d in theirs] == [o.class_name for o in mine]
    for d, o in zip(theirs, mine, strict=True):
        assert d.bbox_xyxy == pytest.approx(o.bbox_xyxy, abs=1e-4)
        assert d.confidence == pytest.approx(o.confidence, abs=1e-6)


@pytest.mark.slow
def test_exported_block_runs_through_pipeline_config_and_backend(tmp_path, helpers, config):
    import yaml
    from cv_module.config import load_config as pavel_config

    torch.manual_seed(0)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(build_detector(config), checkpoint, config, {"run_id": "synthetic"})
    export = tmp_path / "export"
    export_for_pavel(config, checkpoint, 0.1, export)
    block = yaml.safe_load((export / "microorganism_block.yaml").read_text())
    (export / "models" / "floc.pt").write_bytes(b"stub")
    block["models"]["floc_segmentation"] = {"backend": "ultralytics", "weights": "models/floc.pt"}
    (export / "pipeline.yaml").write_text(yaml.safe_dump(block, allow_unicode=True))
    module = pavel_config(export / "pipeline.yaml")
    module.validate(require_models=True)
    assert module.microorganism_classes == config.classes.names
    upstream = TorchvisionCocoModel(
        module.microorganism_model,
        module.microorganism_coco_annotations,
        module.microorganism_architecture,
        module.device,
    )

    class FileBackend(Backend):
        def detect(self, image_bgr):
            return upstream.detect(
                image_bgr,
                confidence=module.microorganism_confidence,
                max_detections=module.max_detections,
            )

        def model_classes(self):
            return {"floc"}, upstream.classes

    source = tmp_path / "input"
    source.mkdir()
    cv2.imwrite(str(source / "frame.bmp"), helpers.synthetic_image(9, 96, 72))
    pipeline = AnalysisPipeline(module, FileBackend(None))
    assert pipeline.validate_model_classes() == []
    pipeline.run(source, tmp_path / "out")
    document = json.loads((tmp_path / "out" / "microorganism_detections.json").read_text())
    with (tmp_path / "out" / "microorganism_counts.csv").open(encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    names = [o["class"] for o in document[0]["objects"]]
    assert set(names) <= set(config.classes.names)
    assert {k: int(v) for k, v in row.items() if k != "image_id"} == {
        n: names.count(n) for n in config.classes.names
    }


def test_upstream_pipeline_tests_pass_in_this_environment(tmp_path, config):
    import shutil
    import subprocess
    import sys

    copy = tmp_path / "pavel"
    shutil.copytree(config.resolve(config.paths.pavel_pipeline) / "tests", copy / "tests")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(copy / "tests")],
        cwd=copy,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:]

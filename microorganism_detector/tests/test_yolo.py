"""YOLO candidate on an ``ultralytics`` test double.

The tests do not load the real package or YOLO weights; they check the
adapters, the output contract, the dataset export and the pipeline hand-over,
not a YOLO model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sludge_micro.config import load_config
from sludge_micro.geometry import yolo_to_xyxy
from sludge_micro.messages import ProjectError
from sludge_micro.yolo import (
    NO_AUGMENT,
    UltralyticsPredictor,
    export_yolo_dataset,
    load_yolo,
    train_arguments,
)

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS = [
    [1.0, 2.0, 30.0, 20.0, 0.9, 2.0],
    [5.0, 5.0, 40.0, 40.0, 0.2, 0.0],
    [10.0, 10.0, 60.0, 45.0, 0.6, 6.0],
]


@pytest.fixture(scope="module")
def yolo_config():
    from sludge_micro.config import YoloConfig

    config = load_config(ROOT / "configs" / "microorganisms.yaml")
    yolo = YoloConfig(
        weights=Path("outputs/pretrained/yolov8s.pt"),
        image_size=1280,
        ignore_policy="exclude_image",
    )
    model = config.model.model_copy(update={"architecture": "ultralytics_yolo", "yolo": yolo})
    return config.model_copy(update={"model": model})


def weights(helpers, tmp_path, config, names=None):
    return helpers.fake_weights(
        tmp_path / "yolo.pt", list(names or config.classes.names), DETECTIONS
    )


def test_missing_package_is_reported(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(ProjectError) as info:
        load_yolo(tmp_path / "w.pt")
    assert info.value.key == "ultralytics_missing"


def test_predictor_follows_the_common_contract(fake_ultralytics, helpers, tmp_path, yolo_config):
    predictor = UltralyticsPredictor(weights(helpers, tmp_path, yolo_config), yolo_config, 0.5)
    image = helpers.synthetic_image(2, 64, 48)
    copy = image.copy()
    objects = predictor.predict_bgr(image)
    assert np.array_equal(image, copy)
    assert [(o.class_name, o.bbox_xyxy) for o in objects] == [
        ("rotifers", (1.0, 2.0, 30.0, 20.0)),
        ("gastrotrichs", (10.0, 10.0, 60.0, 45.0)),
    ]
    call = fake_ultralytics.calls[-1][1]
    assert call["conf"] == yolo_config.postprocess.min_score
    assert call["imgsz"] == yolo_config.model.yolo.image_size


def test_class_order_of_the_weights_is_checked(fake_ultralytics, helpers, tmp_path, yolo_config):
    shuffled = list(reversed(yolo_config.classes.names))
    with pytest.raises(ProjectError) as info:
        UltralyticsPredictor(weights(helpers, tmp_path, yolo_config, shuffled), yolo_config, 0.5)
    assert info.value.key == "checkpoint_classes_mismatch"


def test_dataset_export_round_trips_boxes_and_skips_ignore_images(tmp_path, helpers, yolo_config):
    import cv2

    root = tmp_path / "proj"
    (root / "img").mkdir(parents=True)
    for name in ("a", "b"):
        cv2.imwrite(str(root / "img" / f"{name}.bmp"), helpers.synthetic_image(1, 64, 48))
    images = pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "path": ["img/a.bmp", "img/b.bmp"],
            "width": [64, 64],
            "height": [48, 48],
        }
    )
    anns = pd.DataFrame(
        {
            "image_id": ["a", "b", "b"],
            "supervision": ["target", "target", "ignore"],
            "label": [3, 1, -100],
            "x1": [4.0, 1.0, 5.0],
            "y1": [6.0, 1.0, 5.0],
            "x2": [20.0, 9.0, 9.0],
            "y2": [30.0, 9.0, 9.0],
        }
    )
    config = yolo_config.model_copy(update={"root": root})
    report = export_yolo_dataset(
        config, images, anns, {"a": "train", "b": "validation"}, root / "y"
    )
    assert report == {"images": {"train": 1, "validation": 0, "test": 0}, "excluded_with_ignore": 1}
    line = (root / "y" / "labels" / "train" / "a.txt").read_text().split()
    assert int(line[0]) == 2
    assert yolo_to_xyxy([float(v) for v in line[1:]], 64, 48) == pytest.approx(
        (4, 6, 20, 30), abs=1e-3
    )
    data = yaml.safe_load((root / "y" / "data.yaml").read_text())
    assert list(data["names"].values()) == list(yolo_config.classes.names)


def test_exported_train_folder_lists_the_run_images(tmp_path, yolo_config):
    from sludge_micro.compare import exported_train_ids

    config = yolo_config.model_copy(update={"root": tmp_path})
    folder = config.resolve(config.paths.outputs_dir) / "yolo" / "yolo_dataset" / "images"
    (folder / "train").mkdir(parents=True)
    (folder / "train" / "arch-0001.jpg").write_bytes(b"")
    (folder / "train" / "arch-0002.png").write_bytes(b"")
    assert exported_train_ids(config, "yolo") == {"arch-0001", "arch-0002"}
    assert exported_train_ids(config, "absent") is None


def test_training_arguments_fix_seed_and_disable_augmentation(yolo_config, tmp_path):
    args = train_arguments(yolo_config, tmp_path / "data.yaml", tmp_path)
    assert args["seed"] == yolo_config.runtime.seed and args["deterministic"] is True
    assert all(args[k] == v for k, v in NO_AUGMENT.items())
    rotated = yolo_config.train.augment.model_copy(update={"rotation90": True})
    config = yolo_config.model_copy(
        update={"train": yolo_config.train.model_copy(update={"augment": rotated})}
    )
    with pytest.raises(ProjectError) as info:
        train_arguments(config, tmp_path / "data.yaml", tmp_path)
    assert info.value.key == "yolo_rotation_unsupported"


def test_pipeline_ultralytics_backend_matches_adapter(
    fake_ultralytics, helpers, tmp_path, yolo_config
):
    import csv
    import json

    import cv2
    from cv_module.backends.ultralytics import UltralyticsModel
    from cv_module.config import load_config as pavel_config
    from cv_module.inference.pipeline import AnalysisPipeline
    from cv_module.inference.types import Segmentation

    from sludge_micro.adapter import export_for_pavel

    export = tmp_path / "export"
    export_for_pavel(yolo_config, weights(helpers, tmp_path, yolo_config), 0.5, export)
    block = yaml.safe_load((export / "microorganism_block.yaml").read_text())
    block["models"]["floc_segmentation"] = {"backend": "ultralytics", "weights": "models/floc.pt"}
    (export / "models" / "floc.pt").write_text("stub")
    (export / "pipeline.yaml").write_text(yaml.safe_dump(block))
    module = pavel_config(export / "pipeline.yaml")
    upstream = UltralyticsModel(module.microorganism_model)
    image = helpers.synthetic_image(3, 64, 48)
    theirs = upstream.detect(
        image,
        image_size=module.image_size,
        confidence=module.microorganism_confidence,
        iou=module.iou_threshold,
        device=module.device,
        max_detections=module.max_detections,
    )
    mine = UltralyticsPredictor(
        export / "models" / "microorganisms_yolo.pt", yolo_config, 0.5
    ).predict_bgr(image)
    assert [(d.class_name, d.bbox_xyxy) for d in theirs] == [
        (o.class_name, o.bbox_xyxy) for o in mine
    ]

    class Backend:
        def segment(self, image_bgr):
            return [Segmentation(np.ones(image_bgr.shape[:2], bool), 0.9, "floc")]

        def detect(self, image_bgr):
            return upstream.detect(
                image_bgr,
                image_size=module.image_size,
                confidence=module.microorganism_confidence,
                iou=module.iou_threshold,
                device=module.device,
                max_detections=module.max_detections,
            )

    source = tmp_path / "in"
    source.mkdir()
    cv2.imwrite(str(source / "f.bmp"), image)
    AnalysisPipeline(module, Backend()).run(source, tmp_path / "out")
    document = json.loads((tmp_path / "out" / "microorganism_detections.json").read_text())
    with (tmp_path / "out" / "microorganism_counts.csv").open(encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert [o["class"] for o in document[0]["objects"]] == ["rotifers", "gastrotrichs"]
    assert row["rotifers"] == "1" and row["gastrotrichs"] == "1" and row["nematoda"] == "0"

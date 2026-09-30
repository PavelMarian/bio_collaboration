"""Output contract, colour conversion, input immutability and counts."""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from sludge_micro.infer import (
    Predictor,
    counts_table,
    discover_images,
    predict_images,
    validated,
    write_outputs,
)
from sludge_micro.messages import ProjectError
from sludge_micro.types import ImagePrediction, ObjectDetection

CLASSES = ("attached_ciliates", "rotifers")


class FixedModel(nn.Module):
    """Stand-in detector that records its input and returns fixed boxes."""

    def __init__(self, boxes, labels, scores):
        super().__init__()
        self.out = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels),
            "scores": torch.tensor(scores),
        }
        self.seen = None

    def forward(self, images):
        self.seen = images[0].clone()
        return [self.out]


def predictor(boxes, labels, scores, confidence=0.5):
    return Predictor(FixedModel(boxes, labels, scores), CLASSES, confidence, torch.device("cpu"))


def test_predictor_converts_bgr_to_rgb_and_keeps_input(helpers):
    image = helpers.synthetic_image(3, 40, 30)
    copy = image.copy()
    p = predictor([[1, 2, 10, 20]], [2], [0.9])
    objects = p.predict_bgr(image)
    assert np.array_equal(image, copy)
    assert torch.allclose(p.model.seen[0], torch.from_numpy(image[:, :, 2] / 255.0).float())
    assert objects == (ObjectDetection("rotifers", (1.0, 2.0, 10.0, 20.0), pytest.approx(0.9)),)


def test_confidence_threshold_filters_detections(helpers):
    p = predictor([[1, 1, 5, 5], [2, 2, 8, 8]], [1, 2], [0.4, 0.6])
    assert [o.class_name for o in p.predict_bgr(helpers.synthetic_image(1, 20, 20))] == ["rotifers"]


@pytest.mark.parametrize(
    ("box", "label", "score"),
    [
        ([0, 0, 25, 5], 1, 0.5),
        ([5, 5, 5, 8], 1, 0.5),
        ([0, 0, 5, 5], 3, 0.5),
        ([0, 0, 5, 5], 1, 1.5),
        ([0, 0, float("nan"), 5], 1, 0.5),
    ],
)
def test_invalid_predictions_are_rejected(box, label, score):
    with pytest.raises(ProjectError) as info:
        validated(np.array(box, dtype=float), label, score, (20, 10), CLASSES)
    assert info.value.key == "invalid_prediction"


def test_input_contract_is_checked():
    p = predictor([], [], [])
    with pytest.raises(ProjectError):
        p.predict_bgr(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ProjectError):
        p.predict_bgr(np.zeros((10, 10, 3), dtype=np.float32))


def prediction(image_id, names):
    objects = tuple(ObjectDetection(n, (0.0, 0.0, 1.0, 1.0), 0.5) for n in names)
    return ImagePrediction(image_id, f"{image_id}.jpg", 10, 10, objects)


def test_counts_equal_objects_and_ignore_order():
    preds = [prediction("a", ["rotifers", "rotifers", "attached_ciliates"]), prediction("b", [])]
    table = counts_table(preds, CLASSES).set_index("image_id")
    assert table.loc["a"].to_dict() == {"attached_ciliates": 1, "rotifers": 2}
    assert table.loc["b"].to_dict() == {"attached_ciliates": 0, "rotifers": 0}
    shuffled = [prediction("a", ["attached_ciliates", "rotifers", "rotifers"]), prediction("b", [])]
    assert counts_table(shuffled, CLASSES).equals(counts_table(preds, CLASSES))


def test_outputs_follow_the_pipeline_layout(tmp_path):
    preds = [prediction("x/a", ["rotifers"]), prediction("x/b", [])]
    paths = write_outputs(preds, CLASSES, tmp_path)
    document = json.loads(paths["detections"].read_text(encoding="utf-8"))
    assert document[0]["objects"][0] == {
        "class": "rotifers",
        "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
        "confidence": 0.5,
    }
    assert document[1]["objects"] == []
    with paths["counts"].open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["image_id", *CLASSES]
    assert rows[1] == {"image_id": "x/b", "attached_ciliates": "0", "rotifers": "0"}


def test_image_ids_come_from_relative_paths_and_must_be_unique(tmp_path, helpers):
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    for folder in ("d1", "d2"):
        cv2.imwrite(str(tmp_path / folder / "same.bmp"), helpers.synthetic_image(1))
    assert [i for i, _ in discover_images(tmp_path, [".bmp"])] == ["d1/same", "d2/same"]
    cv2.imwrite(str(tmp_path / "d1" / "same.jpg"), helpers.synthetic_image(1))
    with pytest.raises(ProjectError) as info:
        discover_images(tmp_path, [".bmp", ".jpg"])
    assert info.value.key == "image_id_collision"
    items = discover_images(tmp_path / "d2", [".bmp"])
    result = predict_images(predictor([], [], []), items)
    assert result[0].objects == () and result[0].width == 64

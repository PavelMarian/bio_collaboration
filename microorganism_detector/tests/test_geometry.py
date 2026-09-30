"""Box formats, border checks and metamorphic properties of transforms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sludge_micro.geometry import (
    check_box,
    coco_to_xyxy,
    hflip_boxes,
    ioa_matrix,
    iou_matrix,
    polygons_to_xyxy,
    rot90_boxes,
    rot90_size,
    scale_boxes,
    vflip_boxes,
    xyxy_to_coco,
    yolo_polygon_to_xyxy,
    yolo_to_xyxy,
)


def test_coco_and_xyxy_round_trip():
    box = (10.5, 20.0, 30.0, 40.25)
    assert coco_to_xyxy(box) == (10.5, 20.0, 40.5, 60.25)
    assert xyxy_to_coco(coco_to_xyxy(box)) == pytest.approx(box)


def test_yolo_box_and_polygon_are_denormalised():
    assert yolo_to_xyxy([0.5, 0.5, 0.25, 0.5], 200, 100) == (75.0, 25.0, 125.0, 75.0)
    assert yolo_polygon_to_xyxy([0.1, 0.2, 0.3, 0.2, 0.2, 0.6], 100, 50) == pytest.approx(
        (10.0, 10.0, 30.0, 30.0)
    )


def test_polygon_parts_form_one_box():
    parts = [[0, 0, 10, 0, 10, 10], [50, 60, 55, 70, 52, 65]]
    assert polygons_to_xyxy(parts) == (0.0, 0.0, 55.0, 70.0)
    assert polygons_to_xyxy([]) is None


@pytest.mark.parametrize(
    ("box", "issue", "fix"),
    [
        ((1.0, 1.0, 5.0, 5.0), None, None),
        ((-0.5, 0.0, 5.0, 10.5), None, "clipped_to_border"),
        ((-3.0, 0.0, 5.0, 5.0), "outside_image", None),
        ((4.0, 4.0, 4.0, 8.0), "non_positive_area", None),
        ((math.nan, 0.0, 1.0, 1.0), "non_finite", None),
    ],
)
def test_check_box_reports_issue_and_fix(box, issue, fix):
    result = check_box(box, 10, 10, tolerance=1.0)
    assert result.issue == issue
    assert result.fix == fix
    if issue is None:
        x1, y1, x2, y2 = result.box
        assert 0 <= x1 < x2 <= 10 and 0 <= y1 < y2 <= 10


def mark(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Return the bounding box of non-zero pixels as continuous xyxy."""
    ys, xs = np.nonzero(image)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)


@pytest.mark.parametrize("k", [0, 1, 2, 3, 5, -1])
def test_rot90_boxes_follow_image_pixels(k):
    height, width = 7, 11
    image = np.zeros((height, width), dtype=np.uint8)
    image[1:4, 2:9] = 1
    box = np.array([[2, 1, 9, 4]], dtype=float)
    rotated = np.rot90(image, k)
    moved = rot90_boxes(box, width, height, k)
    assert moved[0] == pytest.approx(mark(rotated, box))
    assert rot90_size(width, height, k) == (rotated.shape[1], rotated.shape[0])


def test_rot90_inverse_restores_geometry():
    rng = np.random.default_rng(0)
    boxes = np.sort(rng.uniform(0, 50, size=(20, 2, 2)), axis=1).transpose(0, 2, 1).reshape(20, 4)
    for k in range(4):
        w, h = rot90_size(80, 60, k)
        back = rot90_boxes(rot90_boxes(boxes, 80, 60, k), w, h, 4 - k)
        assert back == pytest.approx(boxes)


def test_flips_are_involutions_and_scale_inverts():
    boxes = np.array([[1.0, 2.0, 5.0, 9.0], [0.0, 0.0, 20.0, 10.0]])
    assert hflip_boxes(hflip_boxes(boxes, 20), 20) == pytest.approx(boxes)
    assert vflip_boxes(vflip_boxes(boxes, 10), 10) == pytest.approx(boxes)
    assert scale_boxes(scale_boxes(boxes, 0.5, 2.0), 2.0, 0.5) == pytest.approx(boxes)


def test_iou_and_ioa_on_known_boxes():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    assert iou_matrix(a, b)[0] == pytest.approx([50 / 150, 0.0])
    assert ioa_matrix(a, b)[0] == pytest.approx([0.5, 0.0])

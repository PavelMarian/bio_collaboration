"""Bounding-box formats and geometric transforms.

Coordinate conventions:

- COCO ``xywh``: top-left corner plus width and height in pixels.
- model ``xyxy``: ``(x_min, y_min, x_max, y_max)`` in pixels, continuous edges,
  so a full image is ``(0, 0, width, height)``.
- YOLO box: ``(cx, cy, w, h)`` normalised by image width and height.
- YOLO polygon and COCO polygon: flat vertex lists; YOLO values are normalised.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from sludge_micro.types import BoxCheck, BoxXYXY


def coco_to_xyxy(box: Sequence[float]) -> BoxXYXY:
    """Convert a COCO ``[x, y, w, h]`` box to ``xyxy``."""
    x, y, w, h = (float(v) for v in box)
    return (x, y, x + w, y + h)


def xyxy_to_coco(box: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert an ``xyxy`` box to COCO ``[x, y, w, h]``."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x1, y1, x2 - x1, y2 - y1)


def yolo_to_xyxy(values: Sequence[float], width: int, height: int) -> BoxXYXY:
    """Convert a normalised YOLO ``cx cy w h`` box to pixel ``xyxy``."""
    cx, cy, w, h = (float(v) for v in values)
    return (
        (cx - w / 2) * width,
        (cy - h / 2) * height,
        (cx + w / 2) * width,
        (cy + h / 2) * height,
    )


def yolo_polygon_to_xyxy(values: Sequence[float], width: int, height: int) -> BoxXYXY:
    """Return the pixel bounds of a normalised YOLO polygon."""
    xs = [float(v) * width for v in values[0::2]]
    ys = [float(v) * height for v in values[1::2]]
    return (min(xs), min(ys), max(xs), max(ys))


def polygons_to_xyxy(polygons: Sequence[Sequence[float]]) -> BoxXYXY | None:
    """Return the joint bounds of all parts of one COCO polygon annotation.

    Parts of one annotation belong to one object, so the box covers all of them.
    """
    xs = [float(v) for part in polygons for v in part[0::2]]
    ys = [float(v) for part in polygons for v in part[1::2]]
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def check_box(box: BoxXYXY, width: int, height: int, tolerance: float) -> BoxCheck:
    """Validate a box against the image and clip only sub-tolerance overflow.

    Args:
        box: Candidate ``xyxy`` box.
        width: Image width in pixels.
        height: Image height in pixels.
        tolerance: Largest overflow in pixels that is clipped with a recorded fix.

    Returns:
        The accepted box with its fix, or ``None`` with the reason of rejection.
    """
    if not all(math.isfinite(v) for v in box):
        return BoxCheck(None, "non_finite", None)
    x1, y1, x2, y2 = box
    overflow = max(-x1, -y1, x2 - width, y2 - height, 0.0)
    if overflow > tolerance:
        return BoxCheck(None, "outside_image", None)
    clipped = (max(x1, 0.0), max(y1, 0.0), min(x2, float(width)), min(y2, float(height)))
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return BoxCheck(None, "non_positive_area", None)
    fix = "clipped_to_border" if overflow > 0 else None
    return BoxCheck(clipped, None, fix)


def rot90_boxes(boxes: np.ndarray, width: int, height: int, k: int) -> np.ndarray:
    """Rotate ``xyxy`` boxes like ``np.rot90(image, k)`` (counter-clockwise).

    Args:
        boxes: Array of shape ``(N, 4)``.
        width: Width of the image before rotation.
        height: Height of the image before rotation.
        k: Number of quarter turns; any integer.

    Returns:
        New array of rotated boxes; width and height swap on odd ``k``.
    """
    out = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    w, h = width, height
    for _ in range(k % 4):
        x1, y1, x2, y2 = out[:, 0].copy(), out[:, 1].copy(), out[:, 2].copy(), out[:, 3].copy()
        out = np.stack([y1, w - x2, y2, w - x1], axis=1)
        w, h = h, w
    return out


def rot90_size(width: int, height: int, k: int) -> tuple[int, int]:
    """Return image ``(width, height)`` after ``k`` quarter turns."""
    return (height, width) if k % 2 else (width, height)


def hflip_boxes(boxes: np.ndarray, width: int) -> np.ndarray:
    """Mirror ``xyxy`` boxes horizontally."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([width - b[:, 2], b[:, 1], width - b[:, 0], b[:, 3]], axis=1)


def vflip_boxes(boxes: np.ndarray, height: int) -> np.ndarray:
    """Mirror ``xyxy`` boxes vertically."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], height - b[:, 3], b[:, 2], height - b[:, 1]], axis=1)


def scale_boxes(boxes: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Scale ``xyxy`` boxes by independent factors along x and y."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return b * np.array([sx, sy, sx, sy], dtype=np.float64)


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Return areas of ``xyxy`` boxes; degenerate boxes give zero."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the pairwise intersection areas of two box sets."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pairwise intersection over union."""
    inter = intersection(a, b)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def ioa_matrix(a: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Return the share of each box in ``a`` covered by each region."""
    inter = intersection(a, regions)
    area = box_area(a)[:, None]
    return np.divide(inter, area, out=np.zeros_like(inter), where=area > 0)

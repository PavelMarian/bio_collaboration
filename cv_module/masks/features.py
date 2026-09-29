from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import cv2
import numpy as np


FEATURE_COLUMNS = (
    "area_px", "compactness", "fractal_dimension", "mean_brightness",
    "std_brightness", "mean_r", "mean_g", "mean_b", "elongation", "area_fraction",
)
FLOC_CSV_COLUMNS = ("image_id", "annotation_id", "row_type", "floc_count", *FEATURE_COLUMNS)


def fractal_dimension(mask: np.ndarray) -> float:
    """Estimate filled-mask complexity with multi-scale box counting.

    Args:
        mask: Two-dimensional binary instance mask.

    Returns:
        Estimated dimension in the range ``[0, 2]``, or ``NaN`` when the mask
        is empty or too small for a stable estimate.
    """
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2 or not binary.any():
        return math.nan
    ys, xs = np.nonzero(binary)
    if xs.size < 8:
        return math.nan
    crop = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scales: list[int] = []
    scale = 1
    while scale <= min(crop.shape):
        scales.append(scale)
        scale *= 2
    counts: list[int] = []
    used_scales: list[int] = []
    for scale in scales:
        height = math.ceil(crop.shape[0] / scale) * scale
        width = math.ceil(crop.shape[1] / scale) * scale
        padded = np.zeros((height, width), dtype=np.uint8)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        occupied = padded.reshape(height // scale, scale, width // scale, scale).any(axis=(1, 3))
        count = int(occupied.sum())
        if count >= 2:
            used_scales.append(scale)
            counts.append(count)
    if len(counts) < 3:
        return math.nan
    slope = np.polyfit(-np.log(np.asarray(used_scales, dtype=float)), np.log(counts), 1)[0]
    return float(np.clip(slope, 0.0, 2.0))


def extract_features(image_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Calculate all required morphometric and color parameters for one mask.

    Args:
        image_bgr: Source image in OpenCV BGR channel order.
        mask: Two-dimensional binary mask matching the source image size.

    Returns:
        Mapping containing area, compactness, fractal dimension, brightness,
        RGB means, elongation, and image-area fraction.

    Raises:
        ValueError: If the mask shape differs from the image or is empty.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image_bgr.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {image_bgr.shape[:2]}")
    area = int(mask.sum())
    if area == 0:
        raise ValueError("Cannot extract features from an empty mask")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    compactness = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else math.nan
    if math.isfinite(compactness):
        compactness = min(1.0, compactness)
    ys, xs = np.nonzero(mask)
    bbox_w, bbox_h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    gray_values = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)[mask].astype(np.float64)
    bgr_values = image_bgr[mask].astype(np.float64)
    return {
        "area_px": float(area),
        "compactness": float(compactness),
        "fractal_dimension": fractal_dimension(mask),
        "mean_brightness": float(gray_values.mean()),
        "std_brightness": float(gray_values.std(ddof=0)),
        "mean_r": float(bgr_values[:, 2].mean()),
        "mean_g": float(bgr_values[:, 1].mean()),
        "mean_b": float(bgr_values[:, 0].mean()),
        "elongation": float(bbox_w / bbox_h),
        "area_fraction": float(area / mask.size),
    }


def instance_row(image_id: str, annotation_id: int, features: dict[str, float]) -> dict[str, object]:
    """Build one normalized per-floc output row.

    Args:
        image_id: Stable identifier of the source image.
        annotation_id: One-based instance identifier within the image.
        features: Values returned by :func:`extract_features`.

    Returns:
        Row ready for aggregation or CSV serialization.
    """
    return {"image_id": image_id, "annotation_id": annotation_id, "row_type": "instance", "floc_count": 1, **features}


def aggregate_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Append mean and median rows for every image.

    Args:
        rows: Instance rows and optional ``_image`` markers for empty images.

    Returns:
        Instance rows followed by mean and median aggregate rows per image.
    """
    rows = list(rows)
    grouped: dict[object, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row["image_id"]]
        if row["row_type"] == "instance":
            grouped[row["image_id"]].append(row)
    output = [row for row in rows if row["row_type"] == "instance"]
    for image_id, image_rows in grouped.items():
        for statistic, reducer in (("mean", np.nanmean), ("median", np.nanmedian)):
            aggregate: dict[str, object] = {
                "image_id": image_id, "annotation_id": "", "row_type": statistic,
                "floc_count": len(image_rows),
            }
            for column in FEATURE_COLUMNS:
                values = np.asarray([float(row[column]) for row in image_rows], dtype=float)
                aggregate[column] = float(reducer(values)) if np.isfinite(values).any() else math.nan
            output.append(aggregate)
    return output

"""Prediction with an explicit input contract and validated output.

Input: OpenCV BGR ``uint8`` image of shape ``HxWx3``. The input array is never
modified. The detector resizes internally and maps boxes back, so returned
boxes are ``(x_min, y_min, x_max, y_max)`` in pixels of the source image with
``0 <= x_min < x_max <= width`` and ``0 <= y_min < y_max <= height``.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from sludge_micro.imaging import bgr_to_rgb_float, check_bgr_contract, read_bgr
from sludge_micro.messages import ProjectError
from sludge_micro.types import ImagePrediction, ObjectDetection

PREDICTION_COLUMNS = ["image_id", "class_name", "x1", "y1", "x2", "y2", "score"]


class Predictor:
    """Run a detector on BGR images and return validated detections."""

    def __init__(
        self, model: nn.Module, classes: Sequence[str], confidence: float, device: torch.device
    ) -> None:
        """Keep the model in evaluation mode on the given device."""
        self.model = model.to(device).eval()
        self.classes = tuple(classes)
        self.confidence = float(confidence)
        self.device = device

    def raw(self, image_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """Return all boxes, labels and scores above the model's own minimum score."""
        check_bgr_contract(image_bgr)
        rgb = bgr_to_rgb_float(image_bgr)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(self.device)
        with torch.inference_mode():
            out = self.model([tensor])[0]
        return {k: out[k].detach().cpu().numpy() for k in ("boxes", "labels", "scores")}

    def predict_bgr(self, image_bgr: np.ndarray) -> tuple[ObjectDetection, ...]:
        """Return accepted detections of one image at the configured confidence."""
        return accept(self.raw(image_bgr), image_bgr.shape, self.classes, self.confidence)


def accept(
    out: dict[str, np.ndarray], shape: tuple[int, ...], classes: Sequence[str], confidence: float
) -> tuple[ObjectDetection, ...]:
    """Validate raw outputs with one-based labels and keep those at the threshold."""
    height, width = shape[:2]
    return tuple(
        validated(box, int(label), float(score), (width, height), classes)
        for box, label, score in zip(out["boxes"], out["labels"], out["scores"], strict=True)
        if score >= confidence
    )


def validated(
    box: np.ndarray, label: int, score: float, size: tuple[int, int], classes: Sequence[str]
) -> ObjectDetection:
    """Build a detection after checking label, geometry and confidence.

    Raises:
        ProjectError: If any value violates the output contract.
    """
    width, height = size
    x1, y1, x2, y2 = (float(v) for v in box)
    ok = (
        1 <= label <= len(classes)
        and all(math.isfinite(v) for v in (x1, y1, x2, y2, score))
        and 0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
        and 0.0 <= score <= 1.0
    )
    if not ok:
        raise ProjectError("invalid_prediction", details=(label, x1, y1, x2, y2, score))
    return ObjectDetection(classes[label - 1], (x1, y1, x2, y2), score)


def raw_frame(image_id: str, out: dict[str, np.ndarray], classes: Sequence[str]) -> pd.DataFrame:
    """Return raw predictions of one image as a table for evaluation."""
    boxes = out["boxes"].reshape(-1, 4)
    return pd.DataFrame(
        {
            "image_id": image_id,
            "class_name": [classes[int(v) - 1] for v in out["labels"]],
            "x1": boxes[:, 0],
            "y1": boxes[:, 1],
            "x2": boxes[:, 2],
            "y2": boxes[:, 3],
            "score": out["scores"].astype(float),
        },
        columns=PREDICTION_COLUMNS,
    )


def discover_images(source: Path, suffixes: Sequence[str]) -> list[tuple[str, Path]]:
    """List images under a file or directory with unique ids from relative paths.

    Raises:
        ProjectError: On unsupported suffixes, empty input or id collisions.
    """
    allowed = {s.lower() for s in suffixes}
    if source.is_file():
        if source.suffix.lower() not in allowed:
            raise ProjectError("unsupported_image", suffix=source.suffix)
        return [(source.stem, source)]
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in allowed)
    if not files:
        raise ProjectError("no_images", path=source)
    seen: dict[str, Path] = {}
    for path in files:
        image_id = path.relative_to(source).with_suffix("").as_posix()
        if image_id in seen:
            raise ProjectError(
                "image_id_collision", first=seen[image_id], second=path, image_id=image_id
            )
        seen[image_id] = path
    return sorted(seen.items())


def predict_images(
    predictor: Predictor, items: Sequence[tuple[str, Path]]
) -> list[ImagePrediction]:
    """Predict every listed image."""
    results = []
    for image_id, path in items:
        image = read_bgr(path)
        objects = predictor.predict_bgr(image)
        results.append(
            ImagePrediction(image_id, path.name, image.shape[1], image.shape[0], objects)
        )
    return results


def detections_document(predictions: Sequence[ImagePrediction]) -> list[dict[str, Any]]:
    """Serialise predictions in the layout of ``microorganism_detections.json``."""
    return [
        {
            "image_id": p.image_id,
            "file_name": p.file_name,
            "objects": [
                {"class": o.class_name, "bbox_xyxy": list(o.bbox_xyxy), "confidence": o.confidence}
                for o in p.objects
            ],
        }
        for p in predictions
    ]


def counts_table(predictions: Sequence[ImagePrediction], classes: Sequence[str]) -> pd.DataFrame:
    """Count accepted objects per class and image, including zero counts."""
    rows = []
    for p in predictions:
        counts = Counter(o.class_name for o in p.objects)
        rows.append({"image_id": p.image_id, **{name: counts.get(name, 0) for name in classes}})
    return pd.DataFrame(rows, columns=["image_id", *classes])


def write_outputs(
    predictions: Sequence[ImagePrediction], classes: Sequence[str], out_dir: Path
) -> dict[str, Path]:
    """Write detections JSON, counts CSV and an internal Parquet table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "detections": out_dir / "microorganism_detections.json",
        "counts": out_dir / "microorganism_counts.csv",
        "table": out_dir / "microorganism_detections.parquet",
    }
    document = detections_document(predictions)
    paths["detections"].write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    table = counts_table(predictions, classes)
    with paths["counts"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table.columns))
        writer.writeheader()
        writer.writerows(table.to_dict(orient="records"))
    flat = [
        {
            "image_id": p.image_id,
            "class_name": o.class_name,
            "x1": o.bbox_xyxy[0],
            "y1": o.bbox_xyxy[1],
            "x2": o.bbox_xyxy[2],
            "y2": o.bbox_xyxy[3],
            "score": o.confidence,
        }
        for p in predictions
        for o in p.objects
    ]
    pd.DataFrame(flat, columns=PREDICTION_COLUMNS).to_parquet(paths["table"], index=False)
    return paths

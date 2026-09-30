"""One-to-one matching of detections to reference objects and F1 metrics.

Matching per image: detections are processed by descending confidence (ties by
detection index). Each takes the unmatched reference object of the same class
with the highest IoU at or above the threshold (ties by the lowest reference
index). Unmatched detections covered by an ignore region at or above the IoA
threshold are ignored; other unmatched detections are false positives.
Because the order is greedy by confidence, raising the confidence threshold
never changes the outcome of the detections that remain, so one matching pass
serves every threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sludge_micro.config import EvaluationConfig
from sludge_micro.geometry import ioa_matrix, iou_matrix

TP, FP, IGNORED = "tp", "fp", "ignored"
BOX = ["x1", "y1", "x2", "y2"]


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching one image."""

    status: tuple[str, ...]
    matched_reference: tuple[bool, ...]


def match_image(
    det: pd.DataFrame, ref: pd.DataFrame, ignore: pd.DataFrame, iou_thr: float, ioa_thr: float
) -> MatchResult:
    """Match detections of one image to reference objects.

    Args:
        det: Detections with ``class_name``, ``score`` and box columns.
        ref: Reference objects with ``class_name`` and box columns.
        ignore: Ignore regions with box columns.
        iou_thr: Minimum IoU for a true positive.
        ioa_thr: Minimum share of a detection inside an ignore region.
    """
    n, m = len(det), len(ref)
    iou = iou_matrix(det[BOX].to_numpy(float), ref[BOX].to_numpy(float)) if n and m else None
    same = det["class_name"].to_numpy()[:, None] == ref["class_name"].to_numpy()[None, :]
    covered = np.zeros(n)
    if n and len(ignore):
        covered = ioa_matrix(det[BOX].to_numpy(float), ignore[BOX].to_numpy(float)).max(axis=1)
    order = np.lexsort((np.arange(n), -det["score"].to_numpy(float))) if n else []
    matched = np.zeros(m, dtype=bool)
    status = [FP] * n
    for d in order:
        if iou is not None:
            ok = np.flatnonzero(~matched & same[d] & (iou[d] >= iou_thr))
            if ok.size:
                matched[ok[np.argmax(iou[d, ok])]] = True
                status[d] = TP
                continue
        if covered[d] >= ioa_thr:
            status[d] = IGNORED
    return MatchResult(tuple(status), tuple(matched.tolist()))


def match_all(
    det: pd.DataFrame, ref: pd.DataFrame, ignore: pd.DataFrame, cfg: EvaluationConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match every image and return annotated copies of detections and references."""
    det_out, ref_out = [], []
    images = sorted(set(det["image_id"]) | set(ref["image_id"]))
    for image_id in images:
        d = det[det["image_id"] == image_id].reset_index(drop=True)
        r = ref[ref["image_id"] == image_id].reset_index(drop=True)
        g = ignore[ignore["image_id"] == image_id]
        result = match_image(d, r, g, cfg.iou_threshold, cfg.ignore_ioa_threshold)
        det_out.append(d.assign(status=list(result.status)))
        ref_out.append(r.assign(matched=list(result.matched_reference)))
    det_cols = [*det.columns, "status"]
    ref_cols = [*ref.columns, "matched"]
    return (
        pd.concat(det_out, ignore_index=True) if det_out else pd.DataFrame(columns=det_cols),
        pd.concat(ref_out, ignore_index=True) if ref_out else pd.DataFrame(columns=ref_cols),
    )


def ratio(num: int, den: int) -> float | None:
    """Return ``num / den`` or ``None`` when the denominator is zero."""
    return num / den if den else None


def class_row(name: str, tp: int, fp: int, support: int) -> dict[str, Any]:
    """Compute precision, recall and F1 of one class from counts.

    F1 is ``2TP / (2TP + FP + FN)``; it is ``None`` only when the class has no
    reference objects and no predictions.
    """
    fn = support - tp
    return {
        "name": name,
        "support": support,
        "predictions": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, support),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def summarise(
    det: pd.DataFrame, ref: pd.DataFrame, classes: tuple[str, ...], confidence: float
) -> dict[str, Any]:
    """Aggregate matched outcomes at one confidence threshold."""
    kept = det[(det["score"] >= confidence) & (det["status"] != IGNORED)]
    rows = []
    for name in classes:
        part = kept[kept["class_name"] == name]
        tp = int((part["status"] == TP).sum())
        rows.append(class_row(name, tp, len(part) - tp, int((ref["class_name"] == name).sum())))
    defined = [r["f1"] for r in rows if r["f1"] is not None]
    return {
        "confidence": confidence,
        "classes": rows,
        "macro_f1": float(np.mean(defined)) if defined else None,
        "macro_f1_classes": [r["name"] for r in rows if r["f1"] is not None],
    }


def confidence_grid(cfg: EvaluationConfig) -> list[float]:
    """Return the configured confidence grid, rounded to avoid float drift."""
    grid = cfg.confidence_grid
    count = round((grid.stop - grid.start) / grid.step) + 1
    return [round(grid.start + i * grid.step, 10) for i in range(count)]


def select_confidence(
    det: pd.DataFrame, ref: pd.DataFrame, classes: tuple[str, ...], cfg: EvaluationConfig
) -> tuple[float, list[dict[str, Any]]]:
    """Choose the grid threshold with the highest macro F1; ties go higher."""
    curve = []
    for value in confidence_grid(cfg):
        result = summarise(det, ref, classes, value)
        curve.append({"confidence": value, "macro_f1": result["macro_f1"]})
    scored = [c for c in curve if c["macro_f1"] is not None]
    if not scored:
        return confidence_grid(cfg)[-1], curve
    best = max(scored, key=lambda c: (c["macro_f1"], c["confidence"]))
    return best["confidence"], curve


def per_frame(rows: pd.DataFrame, image_ids: list[str], classes: tuple[str, ...]) -> pd.DataFrame:
    """Objects per frame and class, with zeros for frames and classes without rows."""
    if rows.empty:
        return pd.DataFrame(0, index=image_ids, columns=list(classes))
    table = rows.groupby(["image_id", "class_name"]).size().unstack(fill_value=0)
    return table.reindex(index=image_ids, columns=list(classes), fill_value=0)


def count_errors(
    det: pd.DataFrame,
    ref: pd.DataFrame,
    image_ids: list[str],
    classes: tuple[str, ...],
    confidence: float,
) -> dict[str, Any]:
    """Per-class count errors of matched detections at one threshold.

    The error of a frame is the counted minus the annotated number of objects of a
    class; every listed frame counts, frames without objects included. ``counted``
    leaves out the detections that the evaluator ignores (mostly inside an ignore
    region, whose true content is unknown); the ``_all`` values keep them, as the
    counters of the Pavel pipeline do.
    """
    shown = det[det["score"] >= confidence]
    truth = per_frame(ref, image_ids, classes)
    kept = per_frame(shown[shown["status"] != IGNORED], image_ids, classes)
    every = per_frame(shown, image_ids, classes)
    error, error_all = kept - truth, every - truth
    return {
        "images": len(image_ids),
        "confidence": confidence,
        "classes": [
            {
                "name": c,
                "annotated": int(truth[c].sum()),
                "counted": int(kept[c].sum()),
                "in_ignore_regions": int(every[c].sum() - kept[c].sum()),
                "mae_per_frame": float(error[c].abs().mean()),
                "bias_per_frame": float(error[c].mean()),
                "mae_per_frame_all": float(error_all[c].abs().mean()),
                "bias_per_frame_all": float(error_all[c].mean()),
            }
            for c in classes
        ],
    }

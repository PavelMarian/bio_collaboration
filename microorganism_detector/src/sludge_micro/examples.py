"""Validation examples of correct detections, misses and false positives.

The saved validation predictions of a run are matched again at the selected
confidence with the protocol evaluator. Examples are chosen deterministically:
correct detections and misses round-robin over classes, false positives by
descending confidence. Annotated frames go to ``outputs/<run>/examples`` and
the list to ``experiments/<run>/examples.json``.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd

from sludge_micro.config import ProjectConfig
from sludge_micro.evaluate import TP, match_all
from sludge_micro.imaging import read_bgr
from sludge_micro.prepare import load_prepared
from sludge_micro.reporting import read_json, relative, write_json
from sludge_micro.runs import references, run_dirs, split_samples

COLOURS = {"correct": (0, 170, 0), "false_positive": (0, 0, 230), "missed": (230, 120, 0)}
BOX = ["x1", "y1", "x2", "y2"]


def outcomes(config: ProjectConfig, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Matched detections and references of the validation split at the chosen threshold."""
    report_dir, out_dir = run_dirs(config, run_id)
    confidence = read_json(report_dir / "metrics.json")["confidence"]
    preds = pd.read_parquet(out_dir / "predictions_validation.parquet")
    _, anns = split_samples(config, "validation")
    target, ignore = references(anns)
    det, ref = match_all(preds[preds["score"] >= confidence], target, ignore, config.evaluation)
    return det, ref, confidence


def round_robin(rows: pd.DataFrame, classes: tuple[str, ...], limit: int) -> pd.DataFrame:
    """Take rows class by class in label order until the limit is reached."""
    queues = {c: rows[rows["class_name"] == c].reset_index(drop=True) for c in classes}
    picked, depth = [], 0
    while len(picked) < limit and any(depth < len(q) for q in queues.values()):
        picked += [q.iloc[depth] for q in queues.values() if depth < len(q)]
        depth += 1
    return pd.DataFrame(picked[:limit], columns=rows.columns)


def pick(
    det: pd.DataFrame, ref: pd.DataFrame, classes: tuple[str, ...], limit: int
) -> pd.DataFrame:
    """Deterministic examples of each outcome type."""
    tp = det[det["status"] == TP].sort_values(
        ["score", "image_id"], ascending=[False, True], kind="stable"
    )
    fp = det[det["status"] == "fp"].sort_values(
        ["score", "image_id"], ascending=[False, True], kind="stable"
    )
    fn = ref[~ref["matched"].astype(bool)].sort_values(["image_id", "x1"], kind="stable")
    parts = [
        round_robin(tp, classes, limit).assign(kind="correct"),
        fp.head(limit).assign(kind="false_positive"),
        round_robin(fn.assign(score=np.nan), classes, limit).assign(kind="missed"),
    ]
    return pd.concat(parts, ignore_index=True)[["kind", "image_id", "class_name", *BOX, "score"]]


def draw(image: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    """Return a copy of the image with outcome boxes and labels."""
    out = image.copy()
    for row in rows.itertuples():
        colour = COLOURS[row.kind]
        x1, y1, x2, y2 = (round(float(v)) for v in (row.x1, row.y1, row.x2, row.y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 3)
        label = row.class_name if np.isnan(row.score) else f"{row.class_name} {row.score:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return out


def run_examples(config: ProjectConfig, run_id: str, limit: int) -> dict[str, Any]:
    """Render validation examples of one run and list them."""
    det, ref, confidence = outcomes(config, run_id)
    examples = pick(det, ref, config.classes.names, limit)
    images, _ = load_prepared(config)
    paths = dict(zip(images["image_id"], images["path"], strict=True))
    folder = run_dirs(config, run_id)[1] / "examples"
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for image_id, rows in examples.groupby("image_id", sort=True):
        target = folder / f"{image_id}.jpg"
        cv2.imwrite(str(target), draw(read_bgr(config.resolve(paths[image_id])), rows))
        files[image_id] = relative(target, config.root)
    records = [
        {
            "kind": r.kind,
            "image_id": r.image_id,
            "class": r.class_name,
            "bbox_xyxy": [round(float(v), 2) for v in (r.x1, r.y1, r.x2, r.y2)],
            "confidence": None if np.isnan(r.score) else round(float(r.score), 4),
            "file": files[r.image_id],
        }
        for r in examples.itertuples()
    ]
    report = {
        "run_id": run_id,
        "split": "validation",
        "confidence": confidence,
        "examples": records,
    }
    write_json(run_dirs(config, run_id)[0] / "examples.json", report)
    return report

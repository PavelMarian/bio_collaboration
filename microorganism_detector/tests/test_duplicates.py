"""Duplicate grouping by pixels and difference hash, and copy agreement."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from sludge_micro.duplicates import duplicate_groups, group_verdicts
from sludge_micro.imaging import dhash, hamming


def inventory(helpers, images):
    rows = []
    for uid, image in images.items():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows.append(
            {
                "image_uid": uid,
                "source_id": uid.split("/")[0],
                "role": "microorganisms",
                "pixel_sha256": str(hash(image.tobytes())),
                "dhash": dhash(gray, 16),
            }
        )
    return pd.DataFrame(rows)


def test_exact_and_near_copies_group_and_distinct_images_do_not(helpers):
    base = helpers.synthetic_image(1, 128, 96)
    noisy = np.clip(base.astype(int) + 3, 0, 255).astype(np.uint8)
    other = helpers.synthetic_image(2, 128, 96)
    images = inventory(helpers, {"a/1": base, "b/1": base.copy(), "a/2": noisy, "a/3": other})
    groups, pairs = duplicate_groups(images, 10)
    by_uid = dict(zip(images["image_uid"], groups, strict=True))
    assert by_uid["a/1"] == by_uid["b/1"] == by_uid["a/2"] == "a/1"
    assert by_uid["a/3"] == "a/3"
    assert all(p["hamming"] <= 10 for p in pairs)


def test_hamming_distance():
    assert hamming("ff00", "ff01") == 1
    assert hamming("0f", "f0") == 8


def annotations(rows):
    cols = ["image_uid", "role", "category", "x1", "y1", "x2", "y2"]
    return pd.DataFrame(rows, columns=cols)


def test_copy_agreement_detects_class_and_geometry_conflicts():
    images = pd.DataFrame(
        {
            "image_uid": ["a/1", "b/1", "a/2", "b/2"],
            "source_id": ["a", "b", "a", "b"],
            "role": ["microorganisms"] * 4,
            "pixel_sha256": ["p1", "p1", "p2", "p2"],
            "dup_group": ["a/1", "a/1", "a/2", "a/2"],
        }
    )
    anns = annotations(
        [
            ("a/1", "target", "rotifers", 0, 0, 10, 10),
            ("b/1", "target", "rotifers", 0, 0, 10, 10.2),
            ("a/2", "target", "rotifers", 0, 0, 10, 10),
            ("b/2", "target", "nematoda", 0, 0, 10, 10),
        ]
    )
    verdicts = {v["group"]: v for v in group_verdicts(images, anns, 0.9)}
    assert verdicts["a/1"]["annotations_agree"] is True
    assert verdicts["a/2"]["annotations_agree"] is False
    assert verdicts["a/1"]["kind"] == "exact"

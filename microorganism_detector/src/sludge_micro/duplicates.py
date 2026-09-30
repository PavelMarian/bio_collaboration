"""Exact and near-duplicate detection with annotation consistency checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from sludge_micro.geometry import iou_matrix

BLOCK = 512


def hash_words(hashes: list[str]) -> np.ndarray:
    """Pack hexadecimal hashes into ``uint64`` words, one row per hash."""
    raw = np.array([np.frombuffer(bytes.fromhex(h), dtype=">u8") for h in hashes])
    return raw.astype(np.uint64)


def near_pairs(hashes: list[str], max_distance: int) -> list[tuple[int, int, int]]:
    """Return index pairs ``(i, j, distance)`` with ``i < j`` within the distance."""
    if len(hashes) < 2:
        return []
    words = hash_words(hashes)
    pairs = []
    for start in range(0, len(words), BLOCK):
        block = words[start : start + BLOCK]
        dist = np.bitwise_count(block[:, None, :] ^ words[None, :, :]).sum(axis=2)
        rows, cols = np.nonzero(dist <= max_distance)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            i = start + r
            if i < c:
                pairs.append((i, c, int(dist[r, c])))
    return pairs


class UnionFind:
    """Disjoint sets over integer indices."""

    def __init__(self, size: int) -> None:
        """Create ``size`` singleton sets."""
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        """Return the representative of ``item``."""
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        """Merge the sets of ``a`` and ``b``; the smaller index represents them."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def duplicate_groups(images: pd.DataFrame, max_distance: int) -> tuple[pd.Series, list[dict]]:
    """Group images linked by equal pixels or close difference hashes.

    Args:
        images: Inventory with ``image_uid``, ``pixel_sha256`` and ``dhash``.
        max_distance: Largest Hamming distance treated as the same scene.

    Returns:
        Group id per row (the smallest ``image_uid`` of the group) and the pair list.
    """
    ok = images[images["dhash"].notna()].sort_values("image_uid", kind="stable")
    uids = ok["image_uid"].tolist()
    finder = UnionFind(len(uids))
    pairs = []
    pixel = ok["pixel_sha256"].tolist()
    for i, j, dist in near_pairs(ok["dhash"].tolist(), max_distance):
        finder.union(i, j)
        pairs.append(
            {"a": uids[i], "b": uids[j], "hamming": dist, "pixel_equal": pixel[i] == pixel[j]}
        )
    members = defaultdict(list)
    for index, uid in enumerate(uids):
        members[finder.find(index)].append(uid)
    group_of = {uid: min(group) for group in members.values() for uid in group}
    groups = images["image_uid"].map(lambda uid: group_of.get(uid, uid))
    return groups, pairs


def target_boxes(annotations: pd.DataFrame, image_uid: str) -> dict[str, np.ndarray]:
    """Return accepted target boxes of one image keyed by category."""
    rows = annotations[(annotations["image_uid"] == image_uid) & (annotations["role"] == "target")]
    rows = rows[rows["x1"].notna()]
    return {
        name: part[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        for name, part in rows.groupby("category", sort=True)
    }


def boxes_agree(a: np.ndarray, b: np.ndarray, min_iou: float) -> bool:
    """Check a one-to-one pairing of two box sets at the IoU threshold."""
    if len(a) != len(b):
        return False
    iou = iou_matrix(a, b)
    used: set[int] = set()
    for row in iou:
        order = [j for j in np.argsort(-row, kind="stable") if j not in used and row[j] >= min_iou]
        if not order:
            return False
        used.add(int(order[0]))
    return True


def copies_agree(annotations: pd.DataFrame, first: str, second: str, min_iou: float) -> bool:
    """Compare target annotations of two pixel-identical copies."""
    a, b = target_boxes(annotations, first), target_boxes(annotations, second)
    if sorted(a) != sorted(b):
        return False
    return all(boxes_agree(a[name], b[name], min_iou) for name in a)


def group_verdicts(
    images: pd.DataFrame, annotations: pd.DataFrame, min_iou: float
) -> list[dict[str, Any]]:
    """Describe every multi-image group and whether its copies agree.

    Members with equal pixels are copies; others are distinct frames of one
    scene. Annotation agreement is checked only between copies from
    microorganism archives, because distinct frames legitimately differ.
    """
    verdicts = []
    for group, part in images.groupby("dup_group", sort=True):
        if len(part) < 2:
            continue
        part = part.sort_values("image_uid", kind="stable")
        copies = part[part["role"] == "microorganisms"].groupby("pixel_sha256", sort=True)
        checks = [
            copies_agree(annotations, uids[0], other, min_iou)
            for uids in (c["image_uid"].tolist() for _, c in copies)
            for other in uids[1:]
        ]
        distinct = part["pixel_sha256"].nunique()
        kind = "exact" if distinct == 1 else ("near" if distinct == len(part) else "mixed")
        verdicts.append(
            {
                "group": group,
                "members": part["image_uid"].tolist(),
                "kind": kind,
                "sources": sorted(part["source_id"].unique().tolist()),
                "annotations_agree": all(checks) if checks else None,
            }
        )
    return verdicts

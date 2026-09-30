"""Links between frames for the non-temporal group split.

Frames are joined when evidence suggests they share a scene, a series or a
sample:

- pixel copies and near-identical frames (audit duplicate groups);
- overlapping fields of view found by ORB keypoint matching with a RANSAC
  similarity transform (``split.grouping.orb_min_inliers``);
- a series of shots of one subject, named ``X`` and ``X (N)`` in one session;
- a sampling point written in the file name (``ч1``, ``ч2``, ``ил``, aeration
  tank numbers, settler, sample numbers) within one session.

A session is a set of archives linked by shared pixel copies or overlapping
fields of view. Archive names alone are not treated as proof of independent
samples or of one sample. Frames without a sampling-point mark are joined per
session when ``unmarked_per_session`` is set, which is the conservative choice.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

import cv2
import numpy as np
import pandas as pd

from sludge_micro.audit import inventory_dir
from sludge_micro.config import GroupingConfig, ProjectConfig
from sludge_micro.duplicates import UnionFind

SERIES = re.compile(r"\s*\(\d+\)\s*$")
AERATION = (
    re.compile(r"(?<!\d)(\d)\s*а(?:эр)?(?![а-я])"),
    re.compile(r"(?<![а-я])а\s*(\d)(?!\d)"),
    re.compile(r"аэр\.?\s*(\d)"),
)
CH = re.compile(r"(?<![а-я])ч\s*([12])(?!\d)|(?<!\d)([12])\s*ч(?![а-я])|чаша\s*([12])")
SLUDGE = re.compile(r"(^|[^а-я])ил([^а-я]|$)")
LEADING_NUMBER = re.compile(r"^(\d)\.")
TRAILING_NUMBER = re.compile(r"[.\s](\d)\.*$")


def normalised(file_name: str) -> str:
    """Lower-case name without the image suffix, with ``ё`` folded to ``е``."""
    return re.sub(r"\.(bmp|jpe?g)$", "", file_name.lower()).replace("ё", "е")


def series_base(name: str) -> str:
    """Name without a trailing copy counter such as `` (2)``."""
    return SERIES.sub("", name)


def sampling_tokens(file_name: str) -> tuple[str, ...]:
    """Sampling-point marks written in a file name, sorted."""
    name, tokens = series_base(normalised(file_name)), set()
    for pattern in AERATION:
        tokens |= {f"aeration_{m.group(1)}" for m in pattern.finditer(name)}
    if re.search(r"(?<![а-я])аэр(?![а-я])", name) and not tokens:
        tokens.add("aeration")
    if "отст" in name:
        tokens.add("settler")
    for m in CH.finditer(name):
        tokens.add("ch" + (m.group(1) or m.group(2) or m.group(3)))
    if SLUDGE.search(name):
        tokens.add("sludge")
    for pattern in (LEADING_NUMBER, TRAILING_NUMBER):
        match = pattern.search(name)
        if match:
            tokens.add(f"number_{match.group(1)}")
    return tuple(sorted(tokens))


def orb_features(config: ProjectConfig, paths: list[str]) -> list[tuple[np.ndarray, Any]]:
    """Keypoint coordinates and ORB descriptors of downscaled grayscale frames."""
    grouping = config.split.grouping
    orb = cv2.ORB_create(nfeatures=grouping.orb_features)
    features = []
    for path in paths:
        gray = cv2.imdecode(np.fromfile(config.resolve(path), np.uint8), cv2.IMREAD_GRAYSCALE)
        height = round(gray.shape[0] * grouping.orb_image_width / gray.shape[1])
        small = cv2.resize(gray, (grouping.orb_image_width, height), interpolation=cv2.INTER_AREA)
        points, descriptors = orb.detectAndCompute(small, None)
        features.append((np.float32([p.pt for p in points]), descriptors))
    return features


def overlap_inliers(a: tuple, b: tuple, grouping: GroupingConfig, seed: int) -> int:
    """RANSAC inliers of a similarity transform between two frames' keypoints."""
    (pa, da), (pb, db) = a, b
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [
        m
        for m, n in (p for p in pairs if len(p) == 2)
        if m.distance < grouping.orb_ratio * n.distance
    ]
    if len(good) < 3:
        return 0
    cv2.setRNGSeed(seed)
    _, inliers = cv2.estimateAffinePartial2D(
        pa[[g.queryIdx for g in good]],
        pb[[g.trainIdx for g in good]],
        method=cv2.RANSAC,
        ransacReprojThreshold=grouping.orb_ransac_px,
    )
    return int(inliers.sum()) if inliers is not None else 0


def overlap_counts(config: ProjectConfig, images: pd.DataFrame) -> pd.DataFrame:
    """RANSAC inlier count of every pair of frames, sorted by image ids."""
    grouping = config.split.grouping
    cv2.setNumThreads(config.runtime.threads)
    table = images.sort_values("image_id", kind="stable")
    features = orb_features(config, table["path"].tolist())
    ids = table["image_id"].tolist()
    rows = [
        {
            "a": ids[i],
            "b": ids[j],
            "inliers": overlap_inliers(features[i], features[j], grouping, config.runtime.seed),
        }
        for i, j in itertools.combinations(range(len(ids)), 2)
    ]
    return pd.DataFrame(rows, columns=["a", "b", "inliers"])


def overlap_links(config: ProjectConfig, counts: pd.DataFrame) -> pd.DataFrame:
    """Pairs of frames whose fields of view overlap, with their inlier counts."""
    kept = counts[counts["inliers"] >= config.split.grouping.orb_min_inliers]
    return kept.reset_index(drop=True)


def join_by(finder: UnionFind, keys: list[list[Any]]) -> None:
    """Join all indices that share any key."""
    first: dict[Any, int] = {}
    for index, values in enumerate(keys):
        for key in values:
            finder.union(first.setdefault(key, index), index)


def sessions(config: ProjectConfig, images: pd.DataFrame, links: pd.DataFrame) -> dict[str, str]:
    """Map each archive to its session: archives joined by copies or overlaps."""
    inventory = pd.read_parquet(inventory_dir(config) / "images.parquet")
    micro = inventory[inventory["role"] == "microorganisms"]
    sources = sorted(micro["source_id"].unique())
    finder, index = UnionFind(len(sources)), {s: i for i, s in enumerate(sources)}
    for _, part in micro.groupby("dup_group"):
        for a, b in itertools.pairwise(sorted(part["source_id"].unique())):
            finder.union(index[a], index[b])
    source_of = dict(zip(images["image_id"], images["source_id"], strict=True))
    for row in links.itertuples():
        finder.union(index[source_of[row.a]], index[source_of[row.b]])
    return {
        s: "+".join(x for x in sources if finder.find(index[x]) == finder.find(index[s]))
        for s in sources
    }


def image_keys(config: ProjectConfig, table: pd.DataFrame) -> list[list[Any]]:
    """Series and sampling-point keys of every frame within its session."""
    grouping = config.split.grouping
    keys = []
    for session, series, tokens in zip(
        table["session"], table["series"], table["tokens"], strict=True
    ):
        own = [("series", session, series)] if grouping.filename_series else []
        if grouping.sampling_point_tokens:
            own += [("point", session, t) for t in tokens]
            if not tokens and grouping.unmarked_per_session:
                own.append(("point", session, "unmarked"))
        keys.append(own)
    return keys


def scene_groups(
    config: ProjectConfig, images: pd.DataFrame, counts: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign every frame to a group of related frames.

    Args:
        config: Configuration with ``split.grouping``.
        images: Prepared images with paths, sources, names and duplicate groups.
        counts: Pair inlier counts from ``overlap_counts``; computed when omitted.

    Returns:
        Per-image table with session, marks and group id (smallest member id),
        and the overlap links that contributed to the groups.
    """
    table = images.sort_values("image_id", kind="stable").reset_index(drop=True)
    links = overlap_links(config, overlap_counts(config, table) if counts is None else counts)
    session_of = sessions(config, table, links)
    table = table[["image_id", "source_id", "file_name", "dup_group"]].assign(
        session=table["source_id"].map(session_of),
        tokens=table["file_name"].map(sampling_tokens),
        series=table["file_name"].map(lambda n: series_base(normalised(n))),
    )
    finder = UnionFind(len(table))
    join_by(finder, [[g] for g in table["dup_group"]])
    index = {image_id: i for i, image_id in enumerate(table["image_id"])}
    for row in links.itertuples():
        finder.union(index[row.a], index[row.b])
    join_by(finder, image_keys(config, table))
    roots = pd.Series([finder.find(i) for i in range(len(table))])
    first = table["image_id"].groupby(roots).transform("min")
    table["group"] = first.to_numpy()
    table["tokens"] = table["tokens"].map(" ".join)
    return table, links

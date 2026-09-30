"""Non-temporal group split, allowed only with a recorded owner approval.

Groups of related frames (``scenes.scene_groups``) are never divided. The
assignment searches, with the configured seed, for the division of groups
whose split shares of images, of objects per class and of images per class
are closest to ``split.target_fractions``. A class missing from a split while
at least ``len(SPLITS)`` groups contain it adds ``ZERO_PENALTY``. The search is
a seeded random order, a greedy fill by image deficit and single-group moves
that lower the cost; the best of ``split.grouping.search_iterations`` restarts
is kept.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from sludge_micro.config import ProjectConfig
from sludge_micro.prepare import load_prepared
from sludge_micro.reporting import write_json, write_parquet
from sludge_micro.scenes import scene_groups

SPLITS = ("train", "validation", "test")
# One missing class outweighs any share deviation, which is at most 2 per dimension.
ZERO_PENALTY = 10.0


def group_vectors(
    table: pd.DataFrame, anns: pd.DataFrame, classes: tuple[str, ...]
) -> pd.DataFrame:
    """Images, objects per class and images per class of every group."""
    target = anns[anns["supervision"] == "target"]
    objects = target.pivot_table(
        index="image_id", columns="class_name", values="x1", aggfunc="count", fill_value=0
    )
    objects = objects.reindex(index=table["image_id"], columns=list(classes), fill_value=0)
    per_image = pd.DataFrame(objects.to_numpy(), columns=[f"obj:{c}" for c in classes])
    per_image = per_image.assign(
        **{f"img:{c}": (objects[c].to_numpy() > 0).astype(int) for c in classes}
    )
    per_image.insert(0, "images", 1)
    per_image["group"] = table["group"].to_numpy()
    return per_image.groupby("group", sort=True).sum()


def split_cost(
    sums: np.ndarray, totals: np.ndarray, fractions: np.ndarray, needs: np.ndarray
) -> float:
    """Share deviation over all dimensions plus penalties for avoidable missing classes."""
    shares = np.divide(sums, totals, out=np.zeros_like(sums, dtype=float), where=totals > 0)
    cost = float(np.abs(shares - fractions[:, None]).sum())
    return cost + ZERO_PENALTY * float(((sums == 0) & needs[None, :]).sum())


def greedy(order: np.ndarray, vectors: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Give each group, in order, to the split with the largest relative image deficit."""
    assignment = np.zeros(len(vectors), dtype=int)
    images = np.zeros(len(fractions))
    target = fractions * vectors[:, 0].sum()
    for g in order:
        split = int(np.argmax((target - images) / target))
        assignment[g] = split
        images[split] += vectors[g, 0]
    return assignment


def improve(
    assignment: np.ndarray, vectors: np.ndarray, cost: Callable[[np.ndarray], float]
) -> tuple[np.ndarray, float]:
    """Move single groups between splits while the cost decreases."""
    sums = np.stack([vectors[assignment == s].sum(axis=0) for s in range(len(SPLITS))])
    best = cost(sums)
    changed = True
    while changed:
        changed = False
        for g in range(len(vectors)):
            for s in range(len(SPLITS)):
                if s == assignment[g]:
                    continue
                trial = sums.copy()
                trial[assignment[g]] -= vectors[g]
                trial[s] += vectors[g]
                value = cost(trial)
                if value < best - 1e-12:
                    sums, best, assignment[g], changed = trial, value, s, True
    return assignment, best


def assign_groups(config: ProjectConfig, vectors: pd.DataFrame) -> tuple[pd.Series, float]:
    """Search the group assignment with the lowest cost for the configured seed."""
    values = vectors.to_numpy(dtype=float)
    fractions = np.array([config.split.target_fractions[s] for s in SPLITS])
    totals = values.sum(axis=0)
    needs = (values > 0).sum(axis=0) >= len(SPLITS)
    needs[0] = False
    cost = lambda sums: split_cost(sums, totals, fractions, needs)  # noqa: E731
    rng = np.random.default_rng(config.runtime.seed)
    best_assignment, best_cost = None, np.inf
    for _ in range(config.split.grouping.search_iterations):
        start = greedy(rng.permutation(len(values)), values, fractions)
        assignment, value = improve(start, values, cost)
        if value < best_cost - 1e-12:
            best_assignment, best_cost = assignment.copy(), value
    return pd.Series([SPLITS[s] for s in best_assignment], index=vectors.index), best_cost


def class_counts(anns: pd.DataFrame, ids: set[str], classes: tuple[str, ...]) -> dict[str, Any]:
    """Objects and images per class, and ignore regions, for a set of images."""
    part = anns[anns["image_id"].isin(ids)]
    target = part[part["supervision"] == "target"]
    return {
        "objects": {c: int((target["class_name"] == c).sum()) for c in classes},
        "images_with_class": {
            c: int(target.loc[target["class_name"] == c, "image_id"].nunique()) for c in classes
        },
        "ignore_regions": int((part["supervision"] == "ignore").sum()),
        "images_with_ignore": int(part.loc[part["supervision"] == "ignore", "image_id"].nunique()),
    }


def split_statistics(config: ProjectConfig, manifest: pd.DataFrame, anns: pd.DataFrame) -> dict:
    """Images, groups and class counts of every split."""
    out = {}
    for split in SPLITS:
        part = manifest[manifest["split"] == split]
        ids = set(part["image_id"])
        out[split] = {"images": len(ids), "groups": int(part["group"].nunique())} | class_counts(
            anns, ids, config.classes.names
        )
    return out


def train_filter_effect(config: ProjectConfig, manifest: pd.DataFrame, anns: pd.DataFrame) -> dict:
    """What leaving out training images with ignore regions removes from train."""
    train = set(manifest.loc[manifest["split"] == "train", "image_id"])
    removed = train & set(anns.loc[anns["supervision"] == "ignore", "image_id"])
    before = class_counts(anns, train, config.classes.names)["objects"]
    lost = class_counts(anns, removed, config.classes.names)["objects"]
    return {
        "enabled": config.train.exclude_ignore_images,
        "train_images": len(train),
        "removed_images": len(removed),
        "objects_before": before,
        "objects_removed": lost,
        "objects_after": {c: before[c] - lost[c] for c in before},
    }


def limitations(stats: dict, effect: dict, classes: tuple[str, ...]) -> list[dict[str, Any]]:
    """Classes without objects in a split, before and after the training filter."""
    found = [
        {"class": c, "split": s, "stage": "split"}
        for s in SPLITS
        for c in classes
        if stats[s]["objects"][c] == 0
    ]
    if effect["enabled"]:
        found += [
            {"class": c, "split": "train", "stage": "train_filter"}
            for c in classes
            if effect["objects_after"][c] == 0 and effect["objects_before"][c] > 0
        ]
    return found


def run_group_split(config: ProjectConfig) -> dict[str, Any]:
    """Build, verify and lock the approved non-temporal group split."""
    from sludge_micro.archives import sha256_file
    from sludge_micro.split import MANIFEST_COLUMNS, check_independence, lock_path, manifest_path

    images, anns = load_prepared(config)
    table, links = scene_groups(config, images)
    vectors = group_vectors(table, anns, config.classes.names)
    assignment, cost = assign_groups(config, vectors)
    table["split"] = table["group"].map(assignment)
    keep = ["image_id", "file_sha256", "pixel_sha256", "dhash", "captured_at", "captured_at_source"]
    manifest = table.merge(images[keep], on="image_id", how="left")
    manifest = manifest[[*MANIFEST_COLUMNS, "session", "tokens", "series", "file_name"]]
    check_independence(config, manifest)
    write_parquet(manifest_path(config), manifest)
    links_path = manifest_path(config).with_name("scene_links.parquet")
    write_parquet(links_path, links)
    stats = split_statistics(config, manifest, anns)
    effect = train_filter_effect(config, manifest, anns)
    lock = {
        "protocol": config.split.model_dump(mode="json"),
        "data_variant": config.prepare.variant,
        "manifest": manifest_path(config).name,
        "manifest_sha256": sha256_file(manifest_path(config)),
        "scene_links": links_path.name,
        "scene_links_sha256": sha256_file(links_path),
        "groups": int(manifest["group"].nunique()),
        "largest_group": int(manifest["group"].value_counts().max()),
        "sessions": sorted(manifest["session"].unique()),
        "overlap_links": len(links),
        "assignment_cost": round(cost, 6),
        "splits": stats,
        "train_filter": effect,
        "limitations": limitations(stats, effect, config.classes.names),
    }
    write_json(lock_path(config), lock)
    return lock

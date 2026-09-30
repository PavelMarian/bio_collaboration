"""Temporal split protocol and independence checks.

Only confirmed capture times count: EXIF ``DateTimeOriginal`` recorded during
preparation and owner metadata supplied through ``split.owner_metadata``.
Archive names, archive entry dates, file modification times and the COCO
``date_captured`` placeholder never count. Without confirmed times for every
prepared image the split refuses to run.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from sludge_micro.archives import sha256_file
from sludge_micro.config import ProjectConfig
from sludge_micro.duplicates import UnionFind, near_pairs
from sludge_micro.messages import ProjectError
from sludge_micro.prepare import load_prepared
from sludge_micro.reporting import read_json, relative, write_json, write_parquet

SPLITS = ("train", "validation", "test")
MANIFEST_COLUMNS = [
    "image_id",
    "source_id",
    "file_sha256",
    "pixel_sha256",
    "dhash",
    "dup_group",
    "group",
    "captured_at",
    "captured_at_source",
    "split",
]


def split_folder(config: ProjectConfig) -> Path:
    """Folder of the locked split: ``paths.split_dir`` or the experiments folder."""
    return config.resolve(config.paths.split_dir or config.paths.experiments_dir)


def manifest_path(config: ProjectConfig) -> Path:
    """Return the location of the split manifest."""
    return split_folder(config) / "split_manifest.parquet"


def lock_path(config: ProjectConfig) -> Path:
    """Return the location of the split lock record."""
    return split_folder(config) / "split_lock.json"


def owner_times(config: ProjectConfig) -> pd.DataFrame:
    """Read owner metadata with ``image_id``, ``captured_at`` and optional ``series``."""
    path = config.split.owner_metadata
    if path is None:
        return pd.DataFrame(columns=["image_id", "captured_at", "series"])
    full = config.resolve(path)
    if not full.is_file():
        raise ProjectError("file_missing", path=path, action="укажите файл метаданных владельца.")
    table = pd.read_csv(full, dtype=str)
    if "series" not in table:
        table["series"] = None
    return table[["image_id", "captured_at", "series"]]


def confirmed_times(config: ProjectConfig, images: pd.DataFrame) -> pd.DataFrame:
    """Attach confirmed capture times; owner metadata takes precedence over EXIF."""
    allowed = set(config.split.confirmed_time_sources)
    table = images.copy()
    if "exif_datetime_original" not in allowed:
        table["captured_at"], table["captured_at_source"] = None, None
    owner = owner_times(config) if "owner_metadata" in allowed else owner_times(config).iloc[0:0]
    table = table.merge(
        owner.rename(columns={"captured_at": "owner_time"}), on="image_id", how="left"
    )
    has_owner = table["owner_time"].notna()
    table.loc[has_owner, "captured_at"] = table.loc[has_owner, "owner_time"]
    table.loc[has_owner, "captured_at_source"] = "owner_metadata"
    return table.drop(columns="owner_time")


def require_temporal(config: ProjectConfig, table: pd.DataFrame) -> None:
    """Refuse the split without confirmed times for all images or without boundaries."""
    dated = table["captured_at"].notna()
    days = sorted({str(v)[:10] for v in table.loc[dated, "captured_at"]})
    if not dated.all() or len(days) < len(SPLITS):
        raise ProjectError(
            "temporal_metadata_missing", dated=int(dated.sum()), total=len(table), days=len(days)
        )
    if config.split.train_end is None or config.split.validation_end is None:
        raise ProjectError("temporal_boundaries_missing")


def scene_groups(config: ProjectConfig, table: pd.DataFrame) -> pd.Series:
    """Join duplicate groups, near frames and owner series into split groups."""
    finder = UnionFind(len(table))
    keys = ["dup_group", "pixel_sha256"] + (["series"] if "series" in table else [])
    for key in keys:
        first: dict[Any, int] = {}
        for index, value in enumerate(table[key]):
            if pd.notna(value):
                finder.union(first.setdefault(value, index), index)
    for i, j, _ in near_pairs(table["dhash"].tolist(), config.audit.near_duplicate_max_hamming):
        finder.union(i, j)
    ids = table["image_id"].tolist()
    return pd.Series([ids[finder.find(i)] for i in range(len(table))], index=table.index)


def naive(moment: datetime | None) -> datetime | None:
    """Drop time-zone information; capture times are local camera times."""
    return moment.replace(tzinfo=None) if moment is not None else None


def assign_splits(config: ProjectConfig, table: pd.DataFrame) -> pd.Series:
    """Assign splits by time; a group crossing a boundary goes to its latest split."""
    train_end = naive(config.split.train_end)
    val_end = naive(config.split.validation_end)
    moments = table["captured_at"].map(lambda v: naive(datetime.fromisoformat(str(v))))
    rank = moments.map(lambda t: 0 if t <= train_end else (1 if t <= val_end else 2))
    group_rank = rank.groupby(table["group"]).transform("max")
    return group_rank.map(dict(enumerate(SPLITS)))


def check_independence(config: ProjectConfig, manifest: pd.DataFrame) -> None:
    """Refuse splits that share groups, pixel content or near-duplicate frames.

    Raises:
        ProjectError: On any overlap or an empty split.
    """
    for split in SPLITS:
        if not (manifest["split"] == split).any():
            raise ProjectError("split_empty", split=split)
    for key in ("group", "pixel_sha256", "file_sha256"):
        spread = manifest.groupby(key)["split"].nunique()
        if (spread > 1).any():
            raise ProjectError(
                "split_leak", first="-", second="-", key=key, count=int((spread > 1).sum())
            )
    splits = manifest["split"].tolist()
    limit = config.audit.near_duplicate_max_hamming
    crossing = [
        (i, j)
        for i, j, _ in near_pairs(manifest["dhash"].tolist(), limit)
        if splits[i] != splits[j]
    ]
    if crossing:
        i, j = crossing[0]
        raise ProjectError(
            "split_leak", first=splits[i], second=splits[j], key="dhash", count=len(crossing)
        )


def split_summary(manifest: pd.DataFrame) -> dict[str, Any]:
    """Count images and give the capture period of each split."""
    out = {}
    for split in SPLITS:
        part = manifest[manifest["split"] == split]
        out[split] = {
            "images": len(part),
            "groups": int(part["group"].nunique()),
            "period": (
                [min(part["captured_at"]), max(part["captured_at"])]
                if len(part) and part["captured_at"].notna().all()
                else None
            ),
        }
    return out


def run_split(config: ProjectConfig) -> dict[str, Any]:
    """Build, verify and lock the split of the configured protocol.

    Returns:
        The lock record written to ``experiments/split_lock.json``.

    Raises:
        ProjectError: If the configuration reads a split that another stage owns.
    """
    if config.paths.split_dir is not None:
        raise ProjectError("split_owned_elsewhere", path=config.paths.split_dir)
    if config.split.protocol == "group":
        from sludge_micro.group_split import run_group_split

        return run_group_split(config)
    images, _ = load_prepared(config)
    table = confirmed_times(
        config, images.sort_values("image_id", kind="stable").reset_index(drop=True)
    )
    require_temporal(config, table)
    table["group"] = scene_groups(config, table)
    table["split"] = assign_splits(config, table)
    manifest = table[MANIFEST_COLUMNS]
    check_independence(config, manifest)
    write_parquet(manifest_path(config), manifest)
    lock = {
        "protocol": config.split.model_dump(mode="json"),
        "manifest": manifest_path(config).name,
        "manifest_sha256": sha256_file(manifest_path(config)),
        "splits": split_summary(manifest),
    }
    write_json(lock_path(config), lock)
    return lock


def load_split(config: ProjectConfig) -> pd.DataFrame:
    """Return the locked manifest after verifying its checksum.

    Raises:
        ProjectError: If the manifest or lock is missing or the checksum differs.
    """
    path, lock = manifest_path(config), lock_path(config)
    for item in (path, lock):
        if not item.is_file():
            raise ProjectError(
                "file_missing",
                path=relative(item, config.root),
                action="выполните sludge-micro split после решения о протоколе.",
            )
    if sha256_file(path) != read_json(lock)["manifest_sha256"]:
        raise ProjectError("split_lock_mismatch", path=relative(path, config.root))
    return pd.read_parquet(path)


def day_ends(table: pd.DataFrame) -> list[datetime]:
    """Return the last second of every distinct capture day."""
    days = sorted({naive(datetime.fromisoformat(str(v))).date() for v in table["captured_at"]})
    return [datetime.combine(d, datetime.max.time()).replace(microsecond=0) for d in days]


def shares(config: ProjectConfig, table: pd.DataFrame, bounds: tuple[datetime, datetime]) -> dict:
    """Return image shares per split for candidate boundaries."""
    probe = config.model_copy(
        update={
            "split": config.split.model_copy(
                update={"train_end": bounds[0], "validation_end": bounds[1]}
            )
        }
    )
    counts = assign_splits(probe, table).value_counts()
    return {s: float(counts.get(s, 0)) / len(table) for s in SPLITS}


def propose_boundaries(config: ProjectConfig) -> dict[str, Any]:
    """Suggest day boundaries whose split shares are closest to the target fractions.

    Ties go to the earliest boundaries. The owner copies the result into
    ``split.train_end`` and ``split.validation_end`` before building the split.
    """
    images, _ = load_prepared(config)
    table = confirmed_times(
        config, images.sort_values("image_id", kind="stable").reset_index(drop=True)
    )
    probe = config.model_copy(
        update={
            "split": config.split.model_copy(
                update={"train_end": datetime.min, "validation_end": datetime.min}
            )
        }
    )
    require_temporal(probe, table)
    table["group"] = scene_groups(config, table)
    ends, target = day_ends(table), config.split.target_fractions
    best = None
    for i, first in enumerate(ends[:-2]):
        for second in ends[i + 1 : -1]:
            got = shares(config, table, (first, second))
            if min(got.values()) == 0:
                continue
            cost = sum(abs(got[s] - target[s]) for s in SPLITS)
            if best is None or cost < best[0] - 1e-12:
                best = (cost, first, second, got)
    if best is None:
        raise ProjectError("split_empty", split="validation")
    return {
        "train_end": best[1].isoformat(),
        "validation_end": best[2].isoformat(),
        "shares": best[3],
        "target_fractions": dict(target),
        "capture_days": len(ends),
    }

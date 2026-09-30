"""Turn the audited inventory into the prepared microorganism set.

The step applies the configured policies in a fixed order, extracts copies of
the kept images into ``data/processed/extracted/`` and writes
``images.parquet``, ``annotations.parquet`` and ``experiments/prepare_report.json``.
Raw archives are opened read-only.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from sludge_micro.archives import extract_member, open_archive
from sludge_micro.audit import canonical_copies, exif_iso, inventory_dir
from sludge_micro.config import ProjectConfig
from sludge_micro.duplicates import group_verdicts
from sludge_micro.messages import ProjectError
from sludge_micro.reporting import relative, write_json, write_parquet

IGNORE_LABEL = -100
IMAGE_KEEP = [
    "image_id",
    "image_uid",
    "source_id",
    "member",
    "file_name",
    "width",
    "height",
    "file_sha256",
    "pixel_sha256",
    "dhash",
    "dup_group",
    "captured_at",
    "captured_at_source",
]


def load_inventory(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read inventory tables written by the audit.

    Raises:
        ProjectError: If the audit has not been run.
    """
    folder = inventory_dir(config)
    paths = [folder / "images.parquet", folder / "annotations.parquet"]
    for path in paths:
        if not path.is_file():
            raise ProjectError("inventory_missing", path=relative(path, config.root))
    return pd.read_parquet(paths[0]), pd.read_parquet(paths[1])


def assign_ids(images: pd.DataFrame) -> pd.DataFrame:
    """Give each microorganism image a stable, readable and unique identifier."""
    ids = images["source_id"] + "-" + images["source_image_id"].astype(int).map("{:04d}".format)
    if ids.duplicated().any():
        first = ids[ids.duplicated(keep=False)].iloc[0]
        raise ProjectError("image_id_collision", first=first, second=first, image_id=first)
    captured = images["exif_datetime_original"].map(exif_iso)
    return images.assign(
        image_id=ids,
        captured_at=captured,
        captured_at_source=captured.map(lambda v: "exif_datetime_original" if v else None),
    )


def role_counts(annotations: pd.DataFrame) -> pd.DataFrame:
    """Count target, ignore and rejected target boxes per image."""
    ann = annotations.assign(
        target=(annotations["role"] == "target") & annotations["x1"].notna(),
        ignore=(annotations["role"] == "ignore") & annotations["x1"].notna(),
        rejected=(annotations["role"].isin(["target", "ignore"])) & annotations["x1"].isna(),
    )
    return ann.groupby("image_uid")[["target", "ignore", "rejected"]].sum()


ConflictResolver = Callable[[Any], str | None]


def content_reason(config: ProjectConfig, counts: pd.Series) -> str | None:
    """Reason that depends on an image's own annotations, or ``None``."""
    if counts["rejected"] > 0:
        return "invalid_target_box"
    if counts["target"] == 0 and config.prepare.unconfirmed_empty_policy == "exclude":
        return "unconfirmed_empty"
    if counts["ignore"] > 0 and config.prepare.ignore_policy == "exclude_image":
        return "ignore_region"
    return None


def exclusion_reasons(
    config: ProjectConfig,
    images: pd.DataFrame,
    annotations: pd.DataFrame,
    resolve_conflict: ConflictResolver | None = None,
) -> pd.Series:
    """Return the first applicable exclusion reason per image, or ``None``.

    Order: conflicting copies, redundant consistent copy, invalid target box,
    unconfirmed empty image, ignore region under the ``exclude_image`` policy.
    ``resolve_conflict`` exists only for research variants; without it every
    image of a conflicting group is excluded, which is the working policy.
    """
    verdicts = group_verdicts(images, annotations, config.audit.duplicate_match_iou)
    conflicting = {v["group"] for v in verdicts if v["annotations_agree"] is False}
    kept = set(canonical_copies(config, images)["image_uid"])
    counts = role_counts(annotations).reindex(images["image_uid"], fill_value=0)
    reasons = []
    for row, (_, c) in zip(images.itertuples(), counts.iterrows(), strict=True):
        if row.dup_group in conflicting:
            reason = (
                resolve_conflict(row) if resolve_conflict else "conflicting_duplicate_annotations"
            )
        elif row.image_uid not in kept:
            reason = "duplicate_copy"
        else:
            reason = None
        reasons.append(reason or content_reason(config, c))
    return pd.Series(reasons, index=images.index, dtype="object")


def supervision_rows(config: ProjectConfig, annotations: pd.DataFrame) -> pd.DataFrame:
    """Keep accepted target and ignore boxes with their model labels."""
    labels = {t.source_category: (t.name, t.label) for t in config.classes.target}
    rows = annotations[annotations["role"].isin(["target", "ignore"]) & annotations["x1"].notna()]
    mapped = rows["category"].map(lambda c: labels.get(c, (None, IGNORE_LABEL)))
    return pd.DataFrame(
        {
            "ann_uid": rows["ann_uid"],
            "image_uid": rows["image_uid"],
            "source_category": rows["category"],
            "class_name": [m[0] for m in mapped],
            "label": [m[1] for m in mapped],
            "supervision": rows["role"],
            "x1": rows["x1"],
            "y1": rows["y1"],
            "x2": rows["x2"],
            "y2": rows["y2"],
            "bbox_origin": rows["bbox_origin"],
            "fix": rows["fix"],
        }
    ).reset_index(drop=True)


def extract_images(config: ProjectConfig, images: pd.DataFrame) -> list[str]:
    """Extract kept images and return their project-relative paths."""
    base = config.resolve(config.paths.processed_dir) / "extracted"
    raw = config.resolve(config.paths.raw_dir)
    paths = []
    for source_id, part in images.groupby("source_id", sort=True):
        with open_archive(raw / config.source(source_id).archive) as archive:
            for row in part.itertuples():
                target = extract_member(archive, row.member, base / source_id)
                paths.append((row.Index, relative(target, config.root)))
    return [p for _, p in sorted(paths)]


def table_digest(table: pd.DataFrame) -> str:
    """Return a SHA-256 of a table's content independent of the Parquet encoder."""
    records = table.astype(object).where(table.notna(), None).to_dict(orient="records")
    text = json.dumps(records, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unique_empty_reasons(config: ProjectConfig, micro: pd.DataFrame) -> dict[str, int]:
    """Exclusion reasons of unique images without accepted target boxes.

    The audit counts such images on unique pixel content; this table shows
    which exclusion reason each of them received under the fixed order.
    """
    unique = canonical_copies(config, micro)
    empty = unique[unique["target_boxes"] == 0]
    return dict(sorted(Counter(empty["excluded_reason"].fillna("included")).items()))


def prepare_report(
    config: ProjectConfig, micro: pd.DataFrame, kept: pd.DataFrame, sup: pd.DataFrame
) -> dict[str, Any]:
    """Summarise the prepared set without timestamps."""
    target = sup[sup["supervision"] == "target"]
    return {
        "policies": config.prepare.model_dump(mode="json"),
        "images_total": len(micro),
        "images_included": len(kept),
        "excluded_by_reason": dict(sorted(Counter(micro["excluded_reason"].dropna()).items())),
        "included_by_source": dict(sorted(Counter(kept["source_id"]).items())),
        "objects_by_class": {
            n: int((target["class_name"] == n).sum()) for n in config.classes.names
        },
        "images_by_class": {
            n: int(target.loc[target["class_name"] == n, "image_id"].nunique())
            for n in config.classes.names
        },
        "unique_without_target_by_reason": unique_empty_reasons(config, micro),
        "ignore_regions": int((sup["supervision"] == "ignore").sum()),
        "images_with_ignore": int(sup.loc[sup["supervision"] == "ignore", "image_id"].nunique()),
        "images_with_confirmed_time": int(kept["captured_at"].notna().sum()),
        "images_digest": table_digest(kept.drop(columns="path")),
        "annotations_digest": table_digest(sup),
    }


def microorganism_images(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decodable microorganism images with stable ids and their annotations."""
    images, annotations = load_inventory(config)
    usable = (images["role"] == "microorganisms") & images["decode_error"].isna()
    micro = assign_ids(images[usable].reset_index(drop=True))
    return micro, annotations[annotations["image_uid"].isin(micro["image_uid"])]


def build_set(
    config: ProjectConfig,
    micro: pd.DataFrame,
    ann: pd.DataFrame,
    resolve_conflict: ConflictResolver | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply exclusion rules, extract kept images and build supervision rows.

    Returns:
        Images with reasons, kept images with paths, and supervision rows.
    """
    counts = role_counts(ann).reindex(micro["image_uid"], fill_value=0)
    micro = micro.assign(
        excluded_reason=exclusion_reasons(config, micro, ann, resolve_conflict),
        target_boxes=counts["target"].to_numpy(),
    )
    kept = micro[micro["excluded_reason"].isna()].sort_values("image_id", kind="stable")
    kept = kept.reset_index(drop=True)
    kept = kept.assign(path=extract_images(config, kept))[[*IMAGE_KEEP, "path"]]
    sup = supervision_rows(config, ann[ann["image_uid"].isin(kept["image_uid"])])
    sup = sup.merge(kept[["image_uid", "image_id"]], on="image_uid", how="inner")
    sup = sup.sort_values(["image_id", "ann_uid"], kind="stable").reset_index(drop=True)
    return micro, kept, sup


def run_prepare(config: ProjectConfig) -> dict[str, Any]:
    """Apply the working policies, extract kept images and write prepared tables.

    Returns:
        The canonical prepare report.
    """
    micro, kept, sup = build_set(config, *microorganism_images(config))
    processed = config.resolve(config.paths.processed_dir)
    write_parquet(processed / "images.parquet", kept)
    write_parquet(processed / "annotations.parquet", sup)
    excluded = micro[micro["excluded_reason"].notna()][["image_id", "image_uid", "excluded_reason"]]
    write_parquet(processed / "excluded_images.parquet", excluded)
    report = prepare_report(config, micro, kept, sup)
    write_json(config.resolve(config.paths.experiments_dir) / "prepare_report.json", report)
    return report


def load_prepared(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read prepared image and annotation tables of the working set or a variant.

    Raises:
        ProjectError: If the prepare step has not been run.
    """
    processed = config.resolve(config.paths.processed_dir)
    if config.prepare.variant is not None:
        processed = processed / "variants" / config.prepare.variant
    paths = [processed / "images.parquet", processed / "annotations.parquet"]
    for path in paths:
        if not path.is_file():
            raise ProjectError("prepared_missing", path=relative(path, config.root))
    return pd.read_parquet(paths[0]), pd.read_parquet(paths[1])


def image_path(config: ProjectConfig, row: pd.Series) -> Path:
    """Return the absolute path of a prepared image."""
    return config.resolve(row["path"])

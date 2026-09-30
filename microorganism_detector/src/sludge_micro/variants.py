"""Research variants for pixel-identical copies with conflicting annotation.

``exclude_conflicts`` reproduces the working policy. A ``prefer_<source>``
variant keeps, in each conflicting group, the copy from one archive and drops
the other copies; a group without a copy from that archive stays unresolved
and is excluded. Annotations are never merged. All other preparation rules
apply unchanged after the choice. Variants live in ``data/processed/variants``
and are not reference data until biologists confirm the correct annotation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from sludge_micro.archives import sha256_file
from sludge_micro.config import ProjectConfig
from sludge_micro.duplicates import group_verdicts
from sludge_micro.prepare import (
    ConflictResolver,
    build_set,
    microorganism_images,
    table_digest,
)
from sludge_micro.reporting import read_json, relative, write_json, write_parquet

OFFICIAL = "exclude_conflicts"
KEPT = "kept"


def variants_dir(config: ProjectConfig) -> Path:
    """Return the folder of all duplicate-handling variants."""
    return config.resolve(config.paths.processed_dir) / "variants"


def preference_resolver(images: pd.DataFrame, prefer: str) -> ConflictResolver:
    """Keep the copy from ``prefer`` in every conflicting set of equal pixels."""
    sources = images.groupby("pixel_sha256")["source_id"].agg(set)
    chosen = (
        images[images["source_id"] == prefer]
        .sort_values("image_uid", kind="stable")
        .groupby("pixel_sha256")["image_uid"]
        .first()
    )

    def resolve(row: Any) -> str | None:  # noqa: ANN401
        if len(sources[row.pixel_sha256]) < 2:
            return None
        if row.pixel_sha256 not in chosen.index:
            return "unresolved_conflict"
        return None if row.image_uid == chosen[row.pixel_sha256] else "not_preferred_copy"

    return resolve


def conflict_table(
    config: ProjectConfig, micro: pd.DataFrame, annotations: pd.DataFrame
) -> pd.DataFrame:
    """One row per copy in a conflicting group with target counts and the decision."""
    verdicts = group_verdicts(micro, annotations, config.audit.duplicate_match_iou)
    groups = {v["group"] for v in verdicts if v["annotations_agree"] is False}
    part = micro[micro["dup_group"].isin(groups)].sort_values(["dup_group", "image_id"])
    ann = annotations[(annotations["role"] == "target") & annotations["x1"].notna()]
    counts = ann.groupby(["image_uid", "category"]).size().unstack(fill_value=0)
    counts = counts.reindex(
        index=part["image_uid"], columns=list(config.classes.names), fill_value=0
    )
    table = part[["dup_group", "image_id", "source_id", "archive", "member", "source_image_id"]]
    table = table.assign(**{n: counts[n].to_numpy() for n in config.classes.names})
    group_ids = table.groupby("dup_group")["image_id"].transform("min")
    decision = part["excluded_reason"].fillna(KEPT).to_numpy()
    return table.assign(group=group_ids.to_numpy(), decision=decision).drop(columns="dup_group")


def selection_reasons(micro: pd.DataFrame, conflicting: set[str]) -> pd.Series:
    """Why each image is kept or left out.

    Kept images are the single copy of their pixels, the priority copy among
    copies with agreeing annotation, or the copy of the preferred archive in a
    conflicting group. Left-out images carry their exclusion reason.
    """
    copies = micro.groupby("pixel_sha256")["image_uid"].transform("size")
    kept = micro["excluded_reason"].isna()
    reason = pd.Series("single_copy", index=micro.index, dtype="object")
    reason[kept & (copies > 1)] = "priority_among_agreeing_copies"
    reason[kept & (copies > 1) & micro["dup_group"].isin(conflicting)] = "preferred_source"
    reason[~kept] = micro.loc[~kept, "excluded_reason"]
    return reason


def conflicting_groups(config: ProjectConfig, micro: pd.DataFrame, ann: pd.DataFrame) -> set[str]:
    """Duplicate groups whose pixel copies carry different target annotation."""
    verdicts = group_verdicts(micro, ann, config.audit.duplicate_match_iou)
    return {v["group"] for v in verdicts if v["annotations_agree"] is False}


def manifest_table(micro: pd.DataFrame, conflicting: set[str]) -> pd.DataFrame:
    """Kept and excluded images linked to their archive, COCO image id and selection reason."""
    cols = [
        "image_id",
        "image_uid",
        "source_id",
        "archive",
        "member",
        "source_image_id",
        "file_sha256",
        "pixel_sha256",
        "dup_group",
        "excluded_reason",
    ]
    table = micro[cols].assign(
        status=micro["excluded_reason"].isna().map({True: KEPT, False: "excluded"}),
        selection=selection_reasons(micro, conflicting),
        copies=micro.groupby("pixel_sha256")["image_uid"].transform("size"),
    )
    return table.sort_values("image_id", kind="stable").reset_index(drop=True)


def group_outcome(decisions: list[str]) -> str:
    """Summarise what a variant did with one conflicting group."""
    if KEPT in decisions:
        return "copy_kept"
    if "conflicting_duplicate_annotations" in decisions:
        return "group_excluded"
    content = [d for d in decisions if d not in ("not_preferred_copy", "unresolved_conflict")]
    return f"chosen_copy_excluded:{content[0]}" if content else "unresolved"


def group_records(conflicts: pd.DataFrame, classes: tuple[str, ...]) -> list[dict[str, Any]]:
    """Per-group decisions with target counts of every copy."""
    records = []
    for group, part in conflicts.groupby("group", sort=True):
        copies = [
            {
                "image_id": r["image_id"],
                "source_id": r["source_id"],
                "decision": r["decision"],
                "targets": {n: int(r[n]) for n in classes},
            }
            for _, r in part.iterrows()
        ]
        records.append(
            {
                "group": group,
                "outcome": group_outcome([c["decision"] for c in copies]),
                "copies": copies,
            }
        )
    return records


def variant_report(
    config: ProjectConfig, name: str, tables: dict[str, pd.DataFrame], files: dict[str, Any]
) -> dict[str, Any]:
    """Counts, conflict decisions, digests and checksums of one variant."""
    micro, kept, sup, conflicts = (tables[k] for k in ("micro", "kept", "sup", "conflicts"))
    target = sup[sup["supervision"] == "target"]
    groups = group_records(conflicts, config.classes.names)
    return {
        "variant": name,
        "preferred_source": config.duplicate_variants[name],
        "status": "working_policy" if name == OFFICIAL else "research",
        "images_total": len(micro),
        "images_included": len(kept),
        "excluded_by_reason": dict(sorted(Counter(micro["excluded_reason"].dropna()).items())),
        "objects_by_class": {
            n: int((target["class_name"] == n).sum()) for n in config.classes.names
        },
        "images_by_class": {
            n: int(target.loc[target["class_name"] == n, "image_id"].nunique())
            for n in config.classes.names
        },
        "ignore_regions": int((sup["supervision"] == "ignore").sum()),
        "conflict_groups": len(groups),
        "conflict_outcomes": dict(sorted(Counter(g["outcome"] for g in groups).items())),
        "kept_by_selection": dict(
            sorted(
                Counter(
                    selection_reasons(micro, conflicting_groups(config, micro, tables["ann"]))[
                        micro["excluded_reason"].isna()
                    ]
                ).items()
            )
        ),
        "groups": groups,
        "images_digest": table_digest(kept.drop(columns="path")),
        "annotations_digest": table_digest(sup),
        "files": files,
    }


def write_variant(config: ProjectConfig, name: str) -> dict[str, Any]:
    """Build one variant, write its files and return its report."""
    micro, ann = microorganism_images(config)
    micro["archive"] = micro["source_id"].map(lambda s: config.source(s).archive)
    prefer = config.duplicate_variants[name]
    resolver = preference_resolver(micro, prefer) if prefer else None
    micro, kept, sup = build_set(config, micro, ann, resolver)
    conflicts = conflict_table(config, micro, ann)
    folder = variants_dir(config) / name
    outputs = {
        "images.parquet": kept.drop(columns="archive", errors="ignore"),
        "annotations.parquet": sup,
        "manifest.parquet": manifest_table(micro, conflicting_groups(config, micro, ann)),
        "conflict_groups.parquet": conflicts,
    }
    for file, table in outputs.items():
        write_parquet(folder / file, table)
    conflicts.to_csv(folder / "conflict_groups.csv", index=False, encoding="utf-8")
    files = {
        f: {"path": relative(folder / f, config.root), "sha256": sha256_file(folder / f)}
        for f in [*outputs, "conflict_groups.csv"]
    }
    tables = {"micro": micro, "kept": kept, "sup": sup, "conflicts": conflicts, "ann": ann}
    return variant_report(config, name, tables, files)


def run_variants(config: ProjectConfig) -> dict[str, Any]:
    """Build every configured variant and write ``experiments/duplicate_variants.json``."""
    reports = {name: write_variant(config, name) for name in config.duplicate_variants}
    prepared = config.resolve(config.paths.experiments_dir) / "prepare_report.json"
    official = read_json(prepared) if prepared.is_file() else {}
    document = {
        "working_policy": OFFICIAL,
        "working_policy_matches_prepare": (
            reports[OFFICIAL]["images_digest"] == official.get("images_digest")
            and reports[OFFICIAL]["annotations_digest"] == official.get("annotations_digest")
        ),
        "variants": reports,
    }
    write_json(config.resolve(config.paths.experiments_dir) / "duplicate_variants.json", document)
    return document

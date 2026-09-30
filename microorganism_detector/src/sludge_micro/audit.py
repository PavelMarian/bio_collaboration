"""Audit of raw archives: availability, formats, classes, duplicates, metadata.

The audit reads archives without extracting them, writes inventory tables to
``data/processed/inventory/`` and a canonical report to
``experiments/data_audit.json``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from sludge_micro.archives import open_archive, sha256_file
from sludge_micro.checkpoint import inspect_checkpoint
from sludge_micro.config import ProjectConfig
from sludge_micro.duplicates import duplicate_groups, group_verdicts, near_pairs
from sludge_micro.evaluate import match_all, summarise
from sludge_micro.reporting import read_json, relative, write_json, write_parquet
from sludge_micro.sources import SourceTables, read_source

LOG = logging.getLogger(__name__)
FILENAME_STAMP = re.compile(r"^(\d{2})(\d{2})(\d{2})\d{6}_\d+$")


def inventory_dir(config: ProjectConfig) -> Path:
    """Return the directory of inventory tables."""
    return config.resolve(config.paths.processed_dir) / "inventory"


def raw_listing(config: ProjectConfig) -> dict[str, Any]:
    """List declared and undeclared files in the raw directory with checksums."""
    raw_dir = config.resolve(config.paths.raw_dir)
    declared = {s.archive for s in config.sources}
    present = sorted(p.name for p in raw_dir.iterdir() if p.is_file()) if raw_dir.is_dir() else []
    archives = []
    for source in sorted(config.sources, key=lambda s: s.archive):
        path = raw_dir / source.archive
        archives.append(
            {
                "archive": source.archive,
                "source_id": source.id,
                "role": source.role,
                "format": source.format,
                "present": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return {
        "raw_dir": config.paths.raw_dir.as_posix(),
        "archives": archives,
        "undeclared_files": [n for n in present if n not in declared],
    }


def entry_dates(config: ProjectConfig, source_id: str) -> dict[str, Any]:
    """Return the range of archive entry dates of image members."""
    path = config.resolve(config.paths.raw_dir) / config.source(source_id).archive
    with open_archive(path) as archive:
        dates = sorted(
            "{:04d}-{:02d}-{:02d}".format(*m.date_time[:3])
            for m in archive.infolist()
            if Path(m.filename).suffix.lower() in config.audit.image_suffixes
        )
    return {"min": dates[0] if dates else None, "max": dates[-1] if dates else None}


def read_all(config: ProjectConfig) -> dict[str, SourceTables]:
    """Read every present archive into inventory tables."""
    raw_dir = config.resolve(config.paths.raw_dir)
    tables = {}
    for source in config.sources:
        path = raw_dir / source.archive
        if not path.is_file():
            LOG.warning("archive missing: %s", path)
            continue
        with open_archive(path) as archive:
            tables[source.id] = read_source(config, source, archive)
    return tables


def exif_iso(value: object) -> str | None:
    """Convert an EXIF ``YYYY:MM:DD HH:MM:SS`` value to ISO 8601."""
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}", value
    ):
        return None
    day, clock = value.split(" ")
    return f"{day.replace(':', '-')}T{clock}"


def image_summary(config: ProjectConfig, images: pd.DataFrame) -> dict[str, Any]:
    """Summarise formats, sizes and decode problems of one image table."""
    ok = images[images["decode_error"].isna()]
    sizes = Counter(f"{w}x{h}" for w, h in zip(ok["width"], ok["height"], strict=True))
    expected = f"{config.audit.expected_width}x{config.audit.expected_height}"
    declared_mismatch = ok[
        ok["declared_width"].notna()
        & ((ok["declared_width"] != ok["width"]) | (ok["declared_height"] != ok["height"]))
    ]
    return {
        "images": len(images),
        "decode_errors": int(images["decode_error"].notna().sum()),
        "suffixes": dict(
            sorted(Counter(Path(n).suffix.lower() for n in images["file_name"]).items())
        ),
        "sizes": dict(sorted(sizes.items())),
        "images_not_expected_size": int(sum(v for k, v in sizes.items() if k != expected)),
        "declared_size_mismatch": len(declared_mismatch),
        "channels": dict(sorted(Counter(ok["channels"].astype(int)).items())),
        "unreferenced_images": int((~images["referenced"].astype(bool)).sum()),
    }


def annotation_summary(annotations: pd.DataFrame) -> dict[str, Any]:
    """Summarise categories, bbox problems and corrections."""
    return {
        "annotations": len(annotations),
        "by_category": dict(sorted(Counter(annotations["category"].dropna()).items())),
        "by_role": dict(sorted(Counter(annotations["role"].dropna()).items())),
        "issues": dict(sorted(Counter(annotations["issue"].dropna()).items())),
        "fixes": dict(sorted(Counter(annotations["fix"].dropna()).items())),
        "rejected": int(annotations["x1"].isna().sum()),
        "bbox_origin": dict(sorted(Counter(annotations["bbox_origin"].dropna()).items())),
        "bbox_polygon_gap_over_1px": int((annotations["bbox_polygon_gap"].fillna(0) > 1).sum()),
    }


def temporal_summary(images: pd.DataFrame) -> dict[str, Any]:
    """Summarise confirmed and unconfirmed time sources of one image table."""
    iso = images["exif_datetime_original"].map(exif_iso).dropna()
    stamps = images["file_name"].map(lambda n: FILENAME_STAMP.match(Path(n).stem))
    stamp_days = sorted({f"20{m[1]}-{m[2]}-{m[3]}" for m in stamps if m})
    return {
        "exif_datetime_original": len(iso),
        "exif_days": sorted({v[:10] for v in iso}),
        "exif_range": [min(iso), max(iso)] if len(iso) else None,
        "camera_models": dict(sorted(Counter(images["camera_model"].dropna()).items())),
        "coco_date_captured_values": sorted(set(images["date_captured_raw"].dropna())),
        "filename_timestamp_images": int(sum(1 for m in stamps if m)),
        "filename_timestamp_days": len(stamp_days),
        "filename_timestamp_range": [stamp_days[0], stamp_days[-1]] if stamp_days else None,
    }


def source_report(config: ProjectConfig, source_id: str, tables: SourceTables) -> dict[str, Any]:
    """Assemble the report section of one source."""
    return {
        "role": config.source(source_id).role,
        "format": config.source(source_id).format,
        "reader": tables.summary,
        "images": image_summary(config, tables.images),
        "annotations": annotation_summary(tables.annotations),
        "time": temporal_summary(tables.images)
        | {"archive_entry_dates": entry_dates(config, source_id)},
    }


def canonical_copies(config: ProjectConfig, images: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per pixel content, preferring the source with lower priority."""
    priority = {s.id: s.priority for s in config.sources}
    ranked = images.assign(_p=images["source_id"].map(priority))
    ranked = ranked.sort_values(["pixel_sha256", "_p", "image_uid"], kind="stable")
    return ranked.drop_duplicates("pixel_sha256", keep="first").drop(columns="_p")


def per_image_roles(images: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Count accepted annotations of each role per image."""
    accepted = annotations[annotations["x1"].notna()]
    counts = accepted.pivot_table(
        index="image_uid", columns="role", values="ann_uid", aggfunc="count", fill_value=0
    )
    counts = counts.reindex(images["image_uid"], fill_value=0)
    for role in ("target", "ignore", "background", "out_of_scope", "not_detected"):
        if role not in counts:
            counts[role] = 0
    return counts


def box_sizes(rows: pd.DataFrame) -> dict[str, float] | None:
    """Return width and height quantiles of accepted boxes in pixels."""
    if rows.empty:
        return None
    w, h = rows["x2"] - rows["x1"], rows["y2"] - rows["y1"]
    return {
        "width_min": float(w.min()),
        "width_median": float(w.median()),
        "width_max": float(w.max()),
        "height_min": float(h.min()),
        "height_median": float(h.median()),
        "height_max": float(h.max()),
    }


def target_class_rows(config: ProjectConfig, annotations: pd.DataFrame) -> list[dict[str, Any]]:
    """Describe every target class on unique images."""
    out = []
    for target in config.classes.target:
        rows = annotations[annotations["category"] == target.source_category]
        accepted = rows[rows["x1"].notna()]
        out.append(
            {
                "name": target.name,
                "source_category_id": target.source_category_id,
                "label": target.label,
                "objects": len(accepted),
                "rejected": len(rows) - len(accepted),
                "images": int(accepted["image_uid"].nunique()),
                "box_size_px": box_sizes(accepted),
            }
        )
    return out


def microorganism_section(
    config: ProjectConfig, images: pd.DataFrame, annotations: pd.DataFrame
) -> dict[str, Any]:
    """Describe the microorganism archives as one candidate training set."""
    mo_images = images[images["role"] == "microorganisms"]
    mo_ann = annotations[annotations["image_uid"].isin(mo_images["image_uid"])]
    unique = canonical_copies(config, mo_images)
    unique_ann = mo_ann[mo_ann["image_uid"].isin(unique["image_uid"])]
    roles = per_image_roles(unique, unique_ann)
    empty = roles[roles["target"] == 0]
    return {
        "images_all_copies": len(mo_images),
        "images_unique_pixels": len(unique),
        "categories_all_copies": dict(sorted(Counter(mo_ann["category"].dropna()).items())),
        "categories_unique_images": dict(sorted(Counter(unique_ann["category"].dropna()).items())),
        "target_classes_unique_images": target_class_rows(config, unique_ann),
        "images_with_target": int((roles["target"] > 0).sum()),
        "images_without_target": len(empty),
        "images_without_target_with_ignore": int((empty["ignore"] > 0).sum()),
        "images_without_any_annotation": int((roles.sum(axis=1) == 0).sum()),
        "images_with_ignore_regions": int((roles["ignore"] > 0).sum()),
        "images_with_target_and_ignore": int(((roles["target"] > 0) & (roles["ignore"] > 0)).sum()),
        "max_target_objects_per_image": int(roles["target"].max()) if len(roles) else 0,
        "stem_collisions": stem_collisions(mo_images),
    }


def stem_collisions(images: pd.DataFrame) -> dict[str, int]:
    """Count file stems shared by several images; stems serve as image_id upstream."""
    stems = images.assign(stem=images["file_name"].map(lambda n: Path(n).stem))
    shared = stems.groupby("stem")["image_uid"].nunique()
    shared = shared[shared > 1]
    distinct = stems[stems["stem"].isin(shared.index)].groupby("stem")["pixel_sha256"].nunique()
    return {
        "stems_shared": len(shared),
        "images_with_shared_stem": int(shared.sum()),
        "stems_shared_by_different_pixels": int((distinct > 1).sum()),
    }


def duplicates_section(pairs: list[dict], verdicts: list[dict], images: pd.DataFrame) -> dict:
    """Summarise duplicate pairs and groups, including cross-role overlaps."""
    role = dict(zip(images["image_uid"], images["role"], strict=True))
    by_roles = Counter("+".join(sorted({role[p["a"]], role[p["b"]]})) for p in pairs)
    return {
        "pairs": len(pairs),
        "pixel_equal_pairs": sum(1 for p in pairs if p["pixel_equal"]),
        "redundant_file_copies": int(images["file_sha256"].dropna().duplicated().sum()),
        "pairs_by_roles": dict(sorted(by_roles.items())),
        "hamming_histogram": dict(sorted(Counter(str(p["hamming"]) for p in pairs).items())),
        "groups": [v for v in verdicts if "microorganisms" in {role[m] for m in v["members"]}],
        "groups_by_sources": dict(
            sorted(Counter("+".join(v["sources"]) for v in verdicts).items())
        ),
        "groups_by_kind": dict(sorted(Counter(v["kind"] for v in verdicts).items())),
        "conflicting_groups": sum(1 for v in verdicts if v["annotations_agree"] is False),
    }


def spec_comparison(config: ProjectConfig, micro: dict[str, Any]) -> dict[str, Any] | None:
    """Put specification numbers next to audited numbers without adjusting either.

    Returns ``None`` when the configuration names no requirements registry.
    """
    if config.paths.requirements_registry is None:
        return None
    spec = read_json(config.resolve(config.paths.requirements_registry))
    items = {r["id"]: r for r in spec["requirements"]}
    volume, table = items["TZ-B3-02"]["value"], items["TZ-B3-04"]["value"]
    all_c, uniq = micro["categories_all_copies"], micro["categories_unique_images"]
    return {
        "source_requirements": ["TZ-B3-02", "TZ-B3-04"],
        "images": {
            "specification": volume["images"],
            "audit_all_copies": micro["images_all_copies"],
            "audit_unique_pixels": micro["images_unique_pixels"],
        },
        "objects": {
            "specification": volume["objects"],
            "audit_all_copies": sum(all_c.values()),
            "audit_unique_pixels": sum(uniq.values()),
        },
        "classes": [
            {
                "name": row["name"],
                "specification": row["objects"],
                "audit_all_copies": all_c.get(row["name"], 0),
                "audit_unique_pixels": uniq.get(row["name"], 0),
            }
            for row in table
        ],
    }


def external_section(config: ProjectConfig, tables: dict[str, SourceTables]) -> dict[str, Any]:
    """Describe the optional external dataset and its declared class mapping."""
    source_id = config.external.source_id
    if source_id not in tables:
        return {"present": False, "enabled": config.external.enabled}
    ann = tables[source_id].annotations
    mapped = Counter(config.external.class_map.get(c, "unmapped") for c in ann["category"])
    return {
        "present": True,
        "enabled": config.external.enabled,
        "source_id": source_id,
        "categories": dict(sorted(Counter(ann["category"]).items())),
        "mapped_to_targets": dict(sorted(mapped.items())),
        "class_map": dict(sorted(config.external.class_map.items())),
    }


def floc_section(tables: dict[str, SourceTables], sources: list[str]) -> dict[str, Any]:
    """Describe floc-only archives, which carry no microorganism labels."""
    out = {}
    for source_id in sources:
        ann = tables[source_id].annotations
        out[source_id] = {
            "images": len(tables[source_id].images),
            "categories": dict(sorted(Counter(ann["category"].dropna()).items())),
        }
    return out


def reference_section(config: ProjectConfig) -> dict[str, Any] | None:
    """Inspect the received checkpoint and metrics without modifying them.

    Returns ``None`` when the configuration names no received experiment.
    """
    if config.paths.reference_checkpoint is None or config.paths.reference_metrics is None:
        return None
    ck_path = config.resolve(config.paths.reference_checkpoint)
    js_path = config.resolve(config.paths.reference_metrics)
    info = inspect_checkpoint(ck_path, config.model.architecture) if ck_path.is_file() else None
    metrics = read_json(js_path) if js_path.is_file() else None
    history = (metrics or {}).get("history", [])
    return {
        "checkpoint": relative(ck_path, config.root),
        "checkpoint_info": info,
        "metrics_file": relative(js_path, config.root),
        "metrics_sha256": sha256_file(js_path) if js_path.is_file() else None,
        "metrics_keys": sorted(metrics) if metrics else None,
        "reported_f1_macro": (metrics or {}).get("f1_macro"),
        "history_epochs": len(history),
        "class_f1_length": len(history[-1]["class_f1"]) if history else None,
        "class_order": None,
        "split_definition": None,
        "matching_protocol": None,
    }


def target_frame(
    config: ProjectConfig, annotations: pd.DataFrame, uid: str, key: str
) -> pd.DataFrame:
    """Return accepted target boxes of one image under a shared image key."""
    names = {t.source_category: t.name for t in config.classes.target}
    rows = annotations[(annotations["image_uid"] == uid) & (annotations["role"] == "target")]
    rows = rows[rows["x1"].notna()]
    return pd.DataFrame(
        {
            "image_id": key,
            "class_name": rows["category"].map(names),
            "x1": rows["x1"],
            "y1": rows["y1"],
            "x2": rows["x2"],
            "y2": rows["y2"],
            "score": 1.0,
        }
    )


def annotator_agreement(
    config: ProjectConfig, annotations: pd.DataFrame, verdicts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Measure agreement between two annotations of pixel-identical copies.

    The first copy (sorted by ``image_uid``) plays the reference and the second
    the prediction; F1 of this pairing is symmetric and serves as an agreement
    score under the same matching rules as model evaluation.
    """
    micro = {s.id for s in config.sources if s.role == "microorganisms"}
    first, second, origin = [], [], Counter()
    for verdict in verdicts:
        uids = [u for u in verdict["members"] if u.split("/", 1)[0] in micro]
        if verdict["kind"] != "exact" or len(uids) < 2:
            continue
        first.append(target_frame(config, annotations, uids[0], verdict["group"]))
        second.append(target_frame(config, annotations, uids[1], verdict["group"]))
        origin[f"{uids[0].split('/', 1)[0]}>{uids[1].split('/', 1)[0]}"] += 1
    if not first:
        return {"pairs": 0}
    ref, det = pd.concat(first, ignore_index=True), pd.concat(second, ignore_index=True)
    empty = pd.DataFrame(columns=["image_id", "x1", "y1", "x2", "y2"])
    det_m, ref_m = match_all(det, ref, empty, config.evaluation)
    result = summarise(det_m, ref_m, config.classes.names, 0.0)
    return {
        "pairs": len(first),
        "pairs_by_sources": dict(sorted(origin.items())),
        "iou_threshold": config.evaluation.iou_threshold,
        "first_copy_objects": len(ref),
        "second_copy_objects": len(det),
        "matched_objects": int(ref_m["matched"].sum()) if len(ref_m) else 0,
        "classes": result["classes"],
        "macro_f1": result["macro_f1"],
    }


def distance_profile(config: ProjectConfig, images: pd.DataFrame) -> dict[str, int]:
    """Histogram of dHash distances among microorganism images up to twice the threshold."""
    micro = images[(images["role"] == "microorganisms") & images["dhash"].notna()]
    limit = 2 * config.audit.near_duplicate_max_hamming
    pairs = near_pairs(micro.sort_values("image_uid")["dhash"].tolist(), limit)
    return dict(sorted(Counter(str(d) for _, _, d in pairs).items(), key=lambda kv: int(kv[0])))


def finding(code: str, status: str, **evidence: object) -> dict[str, Any]:
    """Build one audit conclusion."""
    return {"id": code, "status": status, "evidence": evidence}


def conclusions(config: ProjectConfig, report: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive audit findings with a status each."""
    micro, dup = report["microorganism_set"], report["duplicates"]
    sources = micro_sources(report)
    days = sorted({d for s in sources for d in s["time"]["exif_days"]})
    dated = sum(s["time"]["exif_datetime_original"] for s in sources)
    weights = config.paths.pretrained_weights
    usable = weights is not None and config.resolve(weights).is_file()
    found = [
        finding("class_map", "ok", archives_checked=len(sources)),
        finding(
            "temporal_split",
            "blocked",
            dated_images=dated,
            images=micro["images_unique_pixels"],
            days=days,
        ),
        finding(
            "duplicates",
            "limitation",
            microorganism_groups=len(dup["groups"]),
            conflicting=dup["conflicting_groups"],
        ),
        finding("unconfirmed_empty_images", "limitation", images=micro["images_without_target"]),
        finding("ignore_regions", "limitation", images=micro["images_with_ignore_regions"]),
        finding("pretrained_weights", "ok" if usable else "blocked", path=weights),
    ]
    return found + reference_findings(report["reference_experiment"])


def reference_findings(reference: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Findings about a received experiment, when the configuration names one."""
    if reference is None:
        return []
    ref = reference["checkpoint_info"] or {}
    return [
        finding(
            "reference_class_order",
            "blocked" if not ref.get("class_names") else "ok",
            classifier_outputs=ref.get("classifier_outputs"),
            class_names=ref.get("class_names"),
        ),
        finding("reference_split", "unknown", split_definition=None),
    ]


def micro_sources(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return per-source sections of microorganism archives."""
    return [s for s in report["sources"].values() if s["role"] == "microorganisms"]


def run_audit(config: ProjectConfig) -> dict[str, Any]:
    """Run the full audit and write its outputs.

    Returns:
        The canonical audit report.
    """
    listing = raw_listing(config)
    tables = read_all(config)
    images = pd.concat([t.images for t in tables.values()], ignore_index=True)
    annotations = pd.concat([t.annotations for t in tables.values()], ignore_index=True)
    groups, pairs = duplicate_groups(images, config.audit.near_duplicate_max_hamming)
    images["dup_group"] = groups.to_numpy()
    verdicts = group_verdicts(images, annotations, config.audit.duplicate_match_iou)
    spec = config.paths.specification
    spec_sha = sha256_file(config.resolve(spec)) if spec is not None else None
    report: dict[str, Any] = {
        "inputs": listing | {"specification_sha256": spec_sha},
        "sources": {sid: source_report(config, sid, t) for sid, t in sorted(tables.items())},
        "microorganism_set": microorganism_section(config, images, annotations),
        "duplicates": duplicates_section(pairs, verdicts, images)
        | {"microorganism_distance_profile": distance_profile(config, images)},
        "annotator_agreement": annotator_agreement(config, annotations, verdicts),
        "external": external_section(config, tables),
        "floc_archives": floc_section(tables, [s.id for s in config.sources if s.role == "floc"]),
        "reference_experiment": reference_section(config),
    }
    report["specification_comparison"] = spec_comparison(config, report["microorganism_set"])
    report["conclusions"] = conclusions(config, report)
    write_parquet(inventory_dir(config) / "images.parquet", images)
    write_parquet(inventory_dir(config) / "annotations.parquet", annotations)
    write_json(config.resolve(config.paths.experiments_dir) / "data_audit.json", report)
    return report

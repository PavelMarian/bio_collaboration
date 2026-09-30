"""Readers that turn raw COCO, YOLO and VOC archives into inventory tables.

Each reader returns a source summary plus two tables: one row per image and
one row per annotation. Readers never modify the archives.
"""

from __future__ import annotations

import json
import logging
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

import pandas as pd

from sludge_micro.config import ProjectConfig, SourceConfig
from sludge_micro.geometry import (
    check_box,
    coco_to_xyxy,
    polygons_to_xyxy,
    yolo_polygon_to_xyxy,
    yolo_to_xyxy,
)
from sludge_micro.imaging import image_facts
from sludge_micro.messages import ProjectError

LOG = logging.getLogger(__name__)

IMAGE_COLUMNS = [
    "image_uid",
    "source_id",
    "role",
    "member",
    "file_name",
    "source_image_id",
    "declared_width",
    "declared_height",
    "width",
    "height",
    "channels",
    "file_sha256",
    "pixel_sha256",
    "dhash",
    "exif_datetime_original",
    "camera_model",
    "date_captured_raw",
    "decode_error",
    "referenced",
]
ANNOTATION_COLUMNS = [
    "ann_uid",
    "image_uid",
    "source_id",
    "source_ann_id",
    "category_id",
    "category",
    "role",
    "raw_bbox",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_origin",
    "issue",
    "fix",
    "bbox_polygon_gap",
    "iscrowd",
]


@dataclass(frozen=True)
class SourceTables:
    """Inventory of one archive."""

    summary: dict[str, Any]
    images: pd.DataFrame
    annotations: pd.DataFrame


def nfc(value: str) -> str:
    """Normalise a name to Unicode NFC for comparisons across file systems."""
    return unicodedata.normalize("NFC", value)


def suffix_index(members: list[str]) -> dict[str, list[str]]:
    """Map every path suffix of every member to the members that end with it."""
    index: dict[str, list[str]] = {}
    for member in members:
        parts = PurePosixPath(nfc(member)).parts
        for start in range(len(parts)):
            index.setdefault("/".join(parts[start:]), []).append(member)
    return index


def image_row(
    config: ProjectConfig, source: SourceConfig, archive: zipfile.ZipFile, member: str
) -> dict[str, Any]:
    """Decode one archive member and return its inventory fields."""
    row: dict[str, Any] = {
        "image_uid": f"{source.id}/{nfc(member)}",
        "source_id": source.id,
        "role": source.role,
        "member": member,
        "file_name": PurePosixPath(member).name,
        "decode_error": None,
    }
    try:
        facts = image_facts(archive.read(member), member, config.audit.dhash_size)
    except ProjectError as exc:
        row["decode_error"] = str(exc)
        return row
    row.update(asdict(facts))
    return row


def image_members(config: ProjectConfig, archive: zipfile.ZipFile) -> list[str]:
    """Return sorted image members with a supported suffix."""
    suffixes = {s.lower() for s in config.audit.image_suffixes}
    names = (m.filename for m in archive.infolist() if not m.is_dir())
    return sorted(n for n in names if PurePosixPath(n).suffix.lower() in suffixes)


def classify_category(config: ProjectConfig, source: SourceConfig, name: str) -> str:
    """Return the configured role of a category or fail on unknown names."""
    role = config.classes.role_of(name)
    if role is None and source.role == "external":
        return "external"
    if role is None:
        raise ProjectError("unknown_category", name=name, source=source.archive)
    return role


def check_category_ids(config: ProjectConfig, source: SourceConfig, cats: dict[int, str]) -> None:
    """Verify that target category ids in the archive equal the configured ids."""
    by_name = {name: cid for cid, name in cats.items()}
    for target in config.classes.target:
        if target.source_category not in by_name:
            raise ProjectError(
                "category_missing", name=target.source_category, source=source.archive
            )
        found = by_name[target.source_category]
        if found != target.source_category_id:
            raise ProjectError(
                "category_schema_mismatch",
                name=target.source_category,
                source=source.archive,
                found=found,
                expected=target.source_category_id,
            )


def coco_box(config: ProjectConfig, ann: dict[str, Any], size: tuple[int, int]) -> dict[str, Any]:
    """Validate the COCO bbox and fall back to polygon bounds when it is invalid."""
    width, height = size
    tol = config.audit.bbox_tolerance_px
    seg = ann.get("segmentation")
    poly = polygons_to_xyxy(seg) if isinstance(seg, list) and seg else None
    raw = ann.get("bbox")
    first = check_box(coco_to_xyxy(raw), width, height, tol) if raw and len(raw) == 4 else None
    gap = None
    if raw and poly and len(raw) == 4:
        gap = max(abs(a - b) for a, b in zip(coco_to_xyxy(raw), poly, strict=True))
    if first is not None and first.box is not None:
        return {
            "box": first.box,
            "origin": "coco_bbox",
            "issue": None,
            "fix": first.fix,
            "gap": gap,
        }
    second = check_box(poly, width, height, tol) if poly else None
    issue = first.issue if first is not None else "bbox_missing"
    if second is not None and second.box is not None:
        fix = "bbox_from_polygon" + (f"+{second.fix}" if second.fix else "")
        return {"box": second.box, "origin": "polygon", "issue": issue, "fix": fix, "gap": gap}
    return {"box": None, "origin": None, "issue": issue, "fix": None, "gap": gap}


def coco_annotation_row(
    config: ProjectConfig, source: SourceConfig, ann: dict[str, Any], ctx: dict[str, Any]
) -> dict[str, Any]:
    """Build one annotation row from a COCO annotation."""
    entry = ctx["images"].get(ann.get("image_id"))
    image_uid, size = entry if entry is not None else (None, None)
    name = ctx["categories"].get(ann.get("category_id"))
    row: dict[str, Any] = {
        "ann_uid": f"{source.id}#{ann.get('id')}",
        "image_uid": image_uid,
        "source_id": source.id,
        "source_ann_id": ann.get("id"),
        "category_id": ann.get("category_id"),
        "category": name,
        "role": classify_category(config, source, name) if name else None,
        "raw_bbox": json.dumps(ann.get("bbox")),
        "iscrowd": int(ann.get("iscrowd", 0)),
    }
    if entry is None:
        return row | {"issue": "orphan_image"}
    if image_uid is None:
        return row | {"issue": "image_file_missing"}
    if name is None:
        return row | {"issue": "unknown_category_id"}
    if size is None:
        return row | {"issue": "image_unreadable"}
    box = coco_box(config, ann, size)
    x1, y1, x2, y2 = box["box"] if box["box"] else (None, None, None, None)
    return row | {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "bbox_origin": box["origin"],
        "issue": box["issue"],
        "fix": box["fix"],
        "bbox_polygon_gap": box["gap"],
    }


def resolve_coco_images(
    config: ProjectConfig, source: SourceConfig, archive: zipfile.ZipFile, doc: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[int, tuple[str | None, tuple[int, int] | None]], Counter]:
    """Match COCO image entries to archive members and decode them."""
    members = image_members(config, archive)
    index = suffix_index(members)
    rows, lookup, problems = [], {}, Counter()
    used: set[str] = set()
    for entry in doc.get("images", []):
        found = index.get(nfc(str(entry.get("file_name", ""))), [])
        if len(found) != 1:
            problems["image_file_missing" if not found else "image_file_ambiguous"] += 1
            lookup[entry.get("id")] = (None, None)
            continue
        row = image_row(config, source, archive, found[0])
        used.add(found[0])
        row |= {
            "source_image_id": entry.get("id"),
            "declared_width": entry.get("width"),
            "declared_height": entry.get("height"),
            "referenced": True,
            "date_captured_raw": json.dumps(entry.get("date_captured")),
        }
        size = None if row["decode_error"] else (row["width"], row["height"])
        lookup[entry.get("id")] = (row["image_uid"], size)
        rows.append(row)
    for member in members:
        if member not in used:
            rows.append(image_row(config, source, archive, member) | {"referenced": False})
            problems["image_without_entry"] += 1
    return rows, lookup, problems


def read_coco(
    config: ProjectConfig, source: SourceConfig, archive: zipfile.ZipFile
) -> SourceTables:
    """Read a COCO archive with exactly one annotation JSON."""
    jsons = [m for m in archive.namelist() if m.lower().endswith(".json")]
    doc = json.loads(archive.read(jsons[0]).decode("utf-8-sig"))
    cats = {int(c["id"]): str(c["name"]) for c in doc.get("categories", [])}
    if source.role == "microorganisms":
        check_category_ids(config, source, cats)
    rows, lookup, problems = resolve_coco_images(config, source, archive, doc)
    ctx = {"images": lookup, "categories": cats}
    anns = [coco_annotation_row(config, source, a, ctx) for a in doc.get("annotations", [])]
    summary = {
        "annotation_files": jsons,
        "categories": {str(k): v for k, v in sorted(cats.items())},
        "info": doc.get("info"),
        "image_entries": len(doc.get("images", [])),
        "annotation_entries": len(doc.get("annotations", [])),
        "problems": dict(sorted(problems.items())),
    }
    return SourceTables(summary, frame(rows, IMAGE_COLUMNS), frame(anns, ANNOTATION_COLUMNS))


def frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """Build a table with a fixed column order."""
    return pd.DataFrame.from_records(rows, columns=columns)


def yolo_line_box(tokens: list[str], size: tuple[int, int]) -> tuple[str, Any]:
    """Classify a YOLO label line as box or polygon and return its pixel bounds."""
    values = [float(t) for t in tokens[1:]]
    if len(values) == 4:
        return "yolo_box", yolo_to_xyxy(values, *size)
    if len(values) >= 6 and len(values) % 2 == 0:
        return "yolo_polygon", yolo_polygon_to_xyxy(values, *size)
    return "malformed", None


def yolo_rows_for_image(
    config: ProjectConfig, source: SourceConfig, text: str, row: dict[str, Any], names: dict
) -> list[dict[str, Any]]:
    """Convert one YOLO label file into annotation rows."""
    out = []
    size = (row["width"], row["height"])
    for number, line in enumerate(x for x in text.splitlines() if x.strip()):
        tokens = line.split()
        name = names.get(int(tokens[0]))
        origin, box = yolo_line_box(tokens, size)
        check = check_box(box, *size, config.audit.bbox_tolerance_px) if box else None
        accepted = check.box if check is not None else None
        out.append(
            {
                "ann_uid": f"{row['image_uid']}#{number}",
                "image_uid": row["image_uid"],
                "source_id": source.id,
                "source_ann_id": number,
                "category_id": int(tokens[0]),
                "category": name,
                "role": classify_category(config, source, name) if name else None,
                "raw_bbox": json.dumps(tokens[1:5]),
                "bbox_origin": origin,
                "x1": accepted[0] if accepted else None,
                "y1": accepted[1] if accepted else None,
                "x2": accepted[2] if accepted else None,
                "y2": accepted[3] if accepted else None,
                "issue": (check.issue if check else "malformed"),
                "fix": check.fix if check else None,
                "iscrowd": 0,
            }
        )
    return out


def read_yolo(
    config: ProjectConfig, source: SourceConfig, archive: zipfile.ZipFile
) -> SourceTables:
    """Read a YOLO archive with ``data.yaml`` and one label file per image."""
    import yaml

    data_yaml = next(m for m in archive.namelist() if m.endswith("data.yaml"))
    meta = yaml.safe_load(archive.read(data_yaml).decode("utf-8"))
    names = {int(k): str(v) for k, v in dict(meta.get("names", {})).items()}
    labels = {PurePosixPath(m).stem: m for m in archive.namelist() if "/labels/" in m}
    rows, anns, problems = [], [], Counter()
    for member in image_members(config, archive):
        row = image_row(config, source, archive, member) | {"referenced": True}
        rows.append(row)
        label = labels.get(PurePosixPath(member).stem)
        if label is None or row["decode_error"]:
            problems["label_file_missing" if label is None else "image_unreadable"] += 1
            continue
        anns.extend(yolo_rows_for_image(config, source, archive.read(label).decode(), row, names))
    summary = {
        "annotation_files": [data_yaml],
        "categories": {str(k): v for k, v in names.items()},
        "label_files": len(labels),
        "problems": dict(sorted(problems.items())),
    }
    return SourceTables(summary, frame(rows, IMAGE_COLUMNS), frame(anns, ANNOTATION_COLUMNS))


def voc_object_row(
    config: ProjectConfig, source: SourceConfig, obj: ET.Element, row: dict[str, Any], n: int
) -> dict[str, Any]:
    """Convert one VOC object; VOC pixel indices are one-based and inclusive."""
    name = obj.findtext("name") or ""
    vals = [float(obj.findtext(f"bndbox/{k}") or "nan") for k in ("xmin", "ymin", "xmax", "ymax")]
    box = (vals[0] - 1, vals[1] - 1, vals[2], vals[3])
    check = check_box(box, row["width"], row["height"], config.audit.bbox_tolerance_px)
    accepted = check.box
    return {
        "ann_uid": f"{row['image_uid']}#{n}",
        "image_uid": row["image_uid"],
        "source_id": source.id,
        "source_ann_id": n,
        "category_id": None,
        "category": name,
        "role": classify_category(config, source, name),
        "raw_bbox": json.dumps(vals),
        "x1": accepted[0] if accepted else None,
        "y1": accepted[1] if accepted else None,
        "x2": accepted[2] if accepted else None,
        "y2": accepted[3] if accepted else None,
        "bbox_origin": "voc_bndbox",
        "issue": check.issue,
        "fix": check.fix,
        "iscrowd": 0,
    }


def read_voc(config: ProjectConfig, source: SourceConfig, archive: zipfile.ZipFile) -> SourceTables:
    """Read a Pascal VOC archive with one XML file per image."""
    xmls = {PurePosixPath(m).stem: m for m in archive.namelist() if m.lower().endswith(".xml")}
    rows, anns, problems, names = [], [], Counter(), Counter()
    for member in image_members(config, archive):
        row = image_row(config, source, archive, member) | {"referenced": True}
        xml_name = xmls.get(PurePosixPath(member).stem)
        if xml_name is None or row["decode_error"]:
            problems["xml_missing" if xml_name is None else "image_unreadable"] += 1
            rows.append(row)
            continue
        root = ET.fromstring(archive.read(xml_name))
        row["declared_width"] = int(root.findtext("size/width") or 0)
        row["declared_height"] = int(root.findtext("size/height") or 0)
        rows.append(row)
        for n, obj in enumerate(root.findall("object")):
            names[obj.findtext("name")] += 1
            anns.append(voc_object_row(config, source, obj, row, n))
    summary = {
        "annotation_files": len(xmls),
        "categories": dict(sorted(names.items())),
        "problems": dict(sorted(problems.items())),
    }
    return SourceTables(summary, frame(rows, IMAGE_COLUMNS), frame(anns, ANNOTATION_COLUMNS))


READERS = {"coco": read_coco, "yolo": read_yolo, "voc": read_voc}


def read_source(
    config: ProjectConfig, source: SourceConfig, archive: zipfile.ZipFile
) -> SourceTables:
    """Dispatch to the reader of the declared annotation format."""
    LOG.info("reading %s (%s)", source.archive, source.format)
    return READERS[source.format](config, source, archive)

"""Readers for COCO, YOLO and VOC archives on synthetic data."""

from __future__ import annotations

import json
import zipfile

import pytest

from sludge_micro.config import SourceConfig
from sludge_micro.messages import ProjectError
from sludge_micro.sources import read_coco, read_voc, read_yolo

COCO = SourceConfig(id="s", archive="s.zip", format="coco", role="microorganisms", priority=0)


def coco_archive(helpers, config, annotations, images=None, files=None):
    images = images or [{"id": 1, "file_name": "x/img.bmp", "width": 64, "height": 48}]
    files = (
        files
        if files is not None
        else {"x/img.bmp": helpers.encode(helpers.synthetic_image(0), ".bmp")}
    )
    doc = helpers.coco_document(config, images, annotations)
    entries = {"annotations/instances_default.json": json.dumps(doc).encode()}
    entries |= {f"images/default/{k}": v for k, v in files.items()}
    return zipfile.ZipFile(helpers.zip_bytes(entries))


def test_coco_rows_have_roles_boxes_and_image_facts(config, helpers):
    anns = [
        helpers.box_ann(1, 1, 3, [1, 2, 10, 20]),
        helpers.box_ann(2, 1, 15, [5, 5, 5, 5]),
        helpers.box_ann(3, 1, 1, [0, 0, 4, 4]),
    ]
    tables = read_coco(config, COCO, coco_archive(helpers, config, anns))
    rows = tables.annotations.set_index("source_ann_id")
    assert list(rows["role"]) == ["target", "ignore", "not_detected"]
    assert tuple(rows.loc[1, ["x1", "y1", "x2", "y2"]]) == (1.0, 2.0, 11.0, 22.0)
    image = tables.images.iloc[0]
    assert (image["width"], image["height"], image["channels"]) == (64, 48, 3)
    assert image["image_uid"] == "s/images/default/x/img.bmp"


def test_invalid_coco_bbox_falls_back_to_polygon(config, helpers):
    ann = helpers.box_ann(1, 1, 3, [1, 2, 10, 20]) | {"bbox": [1, 2, 500, 20]}
    rows = read_coco(config, COCO, coco_archive(helpers, config, [ann])).annotations
    assert rows.loc[0, "issue"] == "outside_image"
    assert rows.loc[0, "fix"] == "bbox_from_polygon"
    assert tuple(rows.loc[0, ["x1", "y1", "x2", "y2"]]) == (1.0, 2.0, 11.0, 22.0)


def test_orphan_annotation_and_missing_file_are_reported(config, helpers):
    images = [
        {"id": 1, "file_name": "x/img.bmp", "width": 64, "height": 48},
        {"id": 2, "file_name": "x/absent.bmp", "width": 64, "height": 48},
    ]
    anns = [helpers.box_ann(1, 9, 3, [1, 1, 5, 5]), helpers.box_ann(2, 2, 3, [1, 1, 5, 5])]
    tables = read_coco(config, COCO, coco_archive(helpers, config, anns, images=images))
    assert list(tables.annotations["issue"]) == ["orphan_image", "image_file_missing"]
    assert tables.summary["problems"] == {"image_file_missing": 1}


def test_category_id_mismatch_stops_the_audit(config, helpers):
    archive = coco_archive(helpers, config, [])
    doc = json.loads(archive.read("annotations/instances_default.json"))
    for category in doc["categories"]:
        if category["name"] == "rotifers":
            category["id"] = 99
    broken = zipfile.ZipFile(helpers.zip_bytes({"annotations/a.json": json.dumps(doc).encode()}))
    with pytest.raises(ProjectError) as info:
        read_coco(config, COCO, broken)
    assert info.value.key == "category_schema_mismatch"


def test_yolo_box_polygon_and_malformed_lines(config, helpers):
    source = SourceConfig(id="y", archive="y.zip", format="yolo", role="floc", priority=0)
    labels = "0 0.5 0.5 0.25 0.5\n0 0.1 0.1 0.2 0.1 0.2 0.3\n0 0.5 0.5\n"
    entries = {
        "e/data.yaml": b"names:\n  0: Active sludge floc\n",
        "e/images/p.jpg": helpers.encode(helpers.synthetic_image(3), ".jpg"),
        "e/labels/p.txt": labels.encode(),
    }
    rows = read_yolo(config, source, zipfile.ZipFile(helpers.zip_bytes(entries))).annotations
    assert list(rows["bbox_origin"]) == ["yolo_box", "yolo_polygon", "malformed"]
    assert tuple(rows.loc[0, ["x1", "y1", "x2", "y2"]]) == (24.0, 12.0, 40.0, 36.0)
    assert rows.loc[2, "issue"] == "malformed"


def test_voc_one_based_inclusive_boxes(config, helpers):
    source = SourceConfig(id="v", archive="v.zip", format="voc", role="external", priority=0)
    xml = (
        "<annotation><size><width>64</width><height>48</height><depth>3</depth></size>"
        "<object><name>Ar</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax>"
        "<ymax>20</ymax></bndbox></object></annotation>"
    )
    entries = {
        "t/images/q.jpg": helpers.encode(helpers.synthetic_image(4), ".jpg"),
        "t/labels/q.xml": xml.encode(),
    }
    tables = read_voc(config, source, zipfile.ZipFile(helpers.zip_bytes(entries)))
    row = tables.annotations.iloc[0]
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (0.0, 0.0, 10.0, 20.0)
    assert row["role"] == "external"
    assert tables.images.iloc[0]["declared_width"] == 64

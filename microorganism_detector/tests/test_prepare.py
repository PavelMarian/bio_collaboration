"""Preparation policies on the synthetic project."""

from __future__ import annotations

import hashlib

import pandas as pd

from sludge_micro.audit import run_audit
from sludge_micro.config import load_config
from sludge_micro.prepare import IGNORE_LABEL, run_prepare


def prepared(project):
    config = load_config(project / "configs" / "microorganisms.yaml")
    run_audit(config)
    return config, run_prepare(config)


def test_policies_exclude_conflicts_invalid_boxes_and_unconfirmed_empty(project):
    _, report = prepared(project)
    assert report["images_total"] == 5
    assert report["excluded_by_reason"] == {
        "conflicting_duplicate_annotations": 2,
        "invalid_target_box": 1,
        "unconfirmed_empty": 1,
    }
    assert report["images_included"] == 1
    assert report["objects_by_class"]["rotifers"] == 1
    assert report["ignore_regions"] == 1
    assert report["images_with_confirmed_time"] == 1


def test_prepared_tables_link_labels_ids_and_extracted_copies(project):
    config, _ = prepared(project)
    images = pd.read_parquet(project / "data" / "processed" / "images.parquet")
    anns = pd.read_parquet(project / "data" / "processed" / "annotations.parquet")
    row = images.iloc[0]
    assert row["image_id"] == "arch_a-0003"
    assert row["captured_at"] == "2024-01-31T10:00:00"
    extracted = config.resolve(row["path"])
    assert extracted.is_relative_to(project / "data" / "processed" / "extracted")
    assert hashlib.sha256(extracted.read_bytes()).hexdigest() == row["file_sha256"]
    labels = dict(zip(anns["supervision"], anns["label"], strict=True))
    assert labels == {"target": 3, "ignore": IGNORE_LABEL}


def test_prepare_is_repeatable(project):
    _, first = prepared(project)
    config = load_config(project / "configs" / "microorganisms.yaml")
    assert run_prepare(config) == first

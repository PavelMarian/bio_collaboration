"""End-to-end audit on a synthetic project: findings, read-only inputs, repeatability."""

from __future__ import annotations

import hashlib

from sludge_micro.audit import run_audit
from sludge_micro.config import load_config


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_finds_copies_conflicts_and_missing_dates(project):
    config = load_config(project / "configs" / "microorganisms.yaml")
    report = run_audit(config)
    micro = report["microorganism_set"]
    assert micro["images_all_copies"] == 5
    assert micro["images_unique_pixels"] == 4
    assert micro["images_without_target"] == 2
    assert micro["images_with_ignore_regions"] == 1
    assert report["duplicates"]["conflicting_groups"] == 1
    temporal = next(c for c in report["conclusions"] if c["id"] == "temporal_split")
    assert temporal["status"] == "blocked"
    assert temporal["evidence"]["dated_images"] == 1
    assert temporal["evidence"]["days"] == ["2024-01-31"]
    source = report["sources"]["arch_a"]
    assert source["annotations"]["issues"] == {"outside_image": 1}
    assert report["annotator_agreement"]["pairs"] == 1
    assert report["annotator_agreement"]["macro_f1"] == 0.0


def test_audit_keeps_raw_archives_unchanged_and_is_repeatable(project):
    config = load_config(project / "configs" / "microorganisms.yaml")
    raw = sorted((project / "data" / "raw").iterdir())
    before = {p.name: digest(p) for p in raw}
    run_audit(config)
    first = (project / "experiments" / "data_audit.json").read_bytes()
    run_audit(config)
    second = (project / "experiments" / "data_audit.json").read_bytes()
    assert first == second
    assert {p.name: digest(p) for p in raw} == before
    assert (project / "data" / "processed" / "inventory" / "images.parquet").is_file()

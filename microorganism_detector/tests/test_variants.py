"""Research variants for conflicting copies on the synthetic project."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from sludge_micro.audit import run_audit
from sludge_micro.config import load_config
from sludge_micro.messages import ProjectError
from sludge_micro.prepare import run_prepare
from sludge_micro.variants import preference_resolver, run_variants


def built(project):
    config = load_config(project / "configs" / "microorganisms.yaml")
    run_audit(config)
    run_prepare(config)
    return config, run_variants(config)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_working_variant_equals_prepared_set(project):
    _, document = built(project)
    assert document["working_policy_matches_prepare"] is True
    official = document["variants"]["exclude_conflicts"]
    assert official["status"] == "working_policy"
    assert official["conflict_outcomes"] == {"group_excluded": 1}


def test_preferred_copy_keeps_its_own_annotation_without_merging(project):
    _, document = built(project)
    for name, source, kept_class in (
        ("prefer_a", "arch_a", "rotifers"),
        ("prefer_b", "arch_b", "attached_ciliates"),
    ):
        report = document["variants"][name]
        assert report["status"] == "research"
        group = report["groups"][0]
        decisions = {c["source_id"]: c["decision"] for c in group["copies"]}
        assert decisions[source] == "kept"
        assert set(decisions.values()) == {"kept", "not_preferred_copy"}
        folder = project / "data" / "processed" / "variants" / name
        anns = pd.read_parquet(folder / "annotations.parquet")
        shared = anns[anns["image_id"].str.endswith("-0001")]
        assert shared["class_name"].tolist() == [kept_class]


def test_manifest_links_archives_and_checksums_match(project):
    config, document = built(project)
    raw = sorted((project / "data" / "raw").iterdir())
    before = {p.name: digest(p) for p in raw}
    for report in document["variants"].values():
        manifest = pd.read_parquet(project / report["files"]["manifest.parquet"]["path"])
        assert set(manifest["archive"]) == {"A.zip", "B.zip"}
        assert manifest["source_image_id"].notna().all()
        assert (manifest["status"] == "kept").sum() == report["images_included"]
        for entry in report["files"].values():
            assert digest(project / entry["path"]) == entry["sha256"]
    run_variants(config)
    assert {p.name: digest(p) for p in raw} == before
    saved = json.loads((project / "experiments" / "duplicate_variants.json").read_text())
    assert saved == json.loads(json.dumps(document))


def test_group_without_preferred_source_stays_unresolved():
    images = pd.DataFrame(
        {
            "image_uid": ["a/1", "b/1", "c/2"],
            "source_id": ["a", "b", "c"],
            "pixel_sha256": ["p", "p", "q"],
        }
    )
    resolve = preference_resolver(images, "c")
    rows = [SimpleNamespace(**r) for r in images.to_dict(orient="records")]
    assert [resolve(r) for r in rows] == ["unresolved_conflict", "unresolved_conflict", None]
    resolve_a = preference_resolver(images, "a")
    assert [resolve_a(r) for r in rows] == [None, "not_preferred_copy", None]


def test_variant_configuration_is_validated(project):
    import yaml

    path = project / "configs" / "microorganisms.yaml"
    data = yaml.safe_load(path.read_text())
    data["duplicate_variants"] = {"prefer_a": "arch_a"}
    path.write_text(yaml.safe_dump(data, allow_unicode=True))
    with pytest.raises(ProjectError):
        load_config(path)
    data["duplicate_variants"] = {"exclude_conflicts": None, "prefer_x": "missing"}
    path.write_text(yaml.safe_dump(data, allow_unicode=True))
    with pytest.raises(ProjectError):
        load_config(path)


def test_manifest_records_why_each_copy_is_kept(project):
    built(project)
    manifest = pd.read_parquet(
        project / "data" / "processed" / "variants" / "prefer_a" / "manifest.parquet"
    )
    selection = dict(zip(manifest["image_id"], manifest["selection"], strict=True))
    assert selection["arch_a-0001"] == "preferred_source"
    assert selection["arch_b-0001"] == "not_preferred_copy"
    assert selection["arch_a-0003"] == "single_copy"
    assert set(manifest.loc[manifest["selection"] == "preferred_source", "copies"]) == {2}


def test_configured_variant_is_read_as_the_prepared_set(project):
    from sludge_micro.prepare import load_prepared

    config, document = built(project)
    variant = config.model_copy(
        update={"prepare": config.prepare.model_copy(update={"variant": "prefer_a"})}
    )
    images, _ = load_prepared(variant)
    assert len(images) == document["variants"]["prefer_a"]["images_included"]
    assert "arch_a-0001" in set(images["image_id"])

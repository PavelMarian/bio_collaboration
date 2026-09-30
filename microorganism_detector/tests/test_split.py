"""Temporal protocol: refusal without dates, grouping, independence, lock."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from sludge_micro.config import load_config
from sludge_micro.messages import ProjectError
from sludge_micro.split import load_split, manifest_path, run_split


def write_prepared(project, rows):
    processed = project / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(processed / "images.parquet", index=False)
    pd.DataFrame(columns=["image_id"]).to_parquet(processed / "annotations.parquet", index=False)


def image(image_id, moment=None, group=None, pixels=None, dhash=None):
    return {
        "image_id": image_id,
        "source_id": "s",
        "file_sha256": f"f-{pixels or image_id}",
        "pixel_sha256": pixels or image_id,
        "dhash": dhash or f"{abs(hash(image_id)) % 16**64:064x}",
        "dup_group": group or image_id,
        "captured_at": moment,
        "captured_at_source": "exif_datetime_original" if moment else None,
    }


def configure(project, owner=None, train_end=None, validation_end=None):
    path = project / "configs" / "microorganisms.yaml"
    data = yaml.safe_load(path.read_text())
    data["split"].update(owner_metadata=owner, train_end=train_end, validation_end=validation_end)
    path.write_text(yaml.safe_dump(data, allow_unicode=True))
    return load_config(path)


def test_split_refuses_without_confirmed_dates(project):
    write_prepared(project, [image("a", "2024-01-31T10:00:00"), image("b"), image("c")])
    with pytest.raises(ProjectError) as info:
        run_split(configure(project, train_end="2024-02-01", validation_end="2024-03-01"))
    assert info.value.key == "temporal_metadata_missing"
    assert info.value.values == {"dated": 1, "total": 3, "days": 1}


def test_split_refuses_without_boundaries(project):
    write_prepared(
        project, [image(i, f"2024-0{m}-01T10:00:00") for i, m in zip("abc", "123", strict=True)]
    )
    with pytest.raises(ProjectError) as info:
        run_split(configure(project))
    assert info.value.key == "temporal_boundaries_missing"


def dated_project(project):
    rows = [
        image("a", "2024-01-05T10:00:00"),
        image("b", "2024-01-06T10:00:00"),
        image("c", "2024-02-05T10:00:00", group="g"),
        image("d", "2024-03-05T10:00:00", group="g"),
        image("e", "2024-02-06T10:00:00"),
        image("f", "2024-03-06T10:00:00"),
    ]
    write_prepared(project, [r | {"captured_at": None, "captured_at_source": None} for r in rows])
    owner = pd.DataFrame(
        {"image_id": [r["image_id"] for r in rows], "captured_at": [r["captured_at"] for r in rows]}
    )
    owner.to_csv(project / "owner.csv", index=False)
    return configure(project, "owner.csv", "2024-01-31T23:59:59", "2024-02-29T23:59:59")


def test_owner_times_split_by_time_and_keep_groups_together(project):
    lock = run_split(dated_project(project))
    manifest = load_split(load_config(project / "configs" / "microorganisms.yaml"))
    split = dict(zip(manifest["image_id"], manifest["split"], strict=True))
    assert split == {
        "a": "train",
        "b": "train",
        "c": "test",
        "d": "test",
        "e": "validation",
        "f": "test",
    }
    assert set(manifest["captured_at_source"]) == {"owner_metadata"}
    assert lock["splits"]["train"]["period"] == ["2024-01-05T10:00:00", "2024-01-06T10:00:00"]


def test_modified_manifest_breaks_the_lock(project):
    config = dated_project(project)
    run_split(config)
    table = pd.read_parquet(manifest_path(config))
    table.loc[0, "split"] = "test"
    table.to_parquet(manifest_path(config), index=False)
    with pytest.raises(ProjectError) as info:
        load_split(config)
    assert info.value.key == "split_lock_mismatch"


def test_pixel_copies_never_cross_splits(project):
    config = dated_project(project)
    rows = pd.read_parquet(project / "data" / "processed" / "images.parquet")
    rows.loc[rows["image_id"] == "f", ["pixel_sha256", "file_sha256"]] = ["a", "f-a"]
    rows.to_parquet(project / "data" / "processed" / "images.parquet", index=False)
    run_split(config)
    manifest = load_split(config)
    split = dict(zip(manifest["image_id"], manifest["split"], strict=True))
    assert split["a"] == split["f"] == "test"


def test_boundary_proposal_uses_day_ends_closest_to_target_shares(project):
    from sludge_micro.split import propose_boundaries

    rows = [image(f"i{d:02d}", f"2024-01-{d:02d}T10:00:00") for d in range(1, 11)]
    write_prepared(project, rows)
    proposal = propose_boundaries(configure(project))
    assert proposal["train_end"] == "2024-01-06T23:59:59"
    assert proposal["validation_end"] == "2024-01-08T23:59:59"
    assert proposal["shares"] == {"train": 0.6, "validation": 0.2, "test": 0.2}
    assert proposal["capture_days"] == 10


def test_boundary_proposal_refuses_without_dates(project):
    from sludge_micro.split import propose_boundaries

    write_prepared(project, [image("a", "2024-01-01T10:00:00"), image("b"), image("c")])
    with pytest.raises(ProjectError) as info:
        propose_boundaries(configure(project))
    assert info.value.key == "temporal_metadata_missing"


def test_a_split_owned_by_another_stage_is_read_there_and_never_rebuilt(config):
    from pathlib import Path

    from sludge_micro.split import lock_path, manifest_path, run_split

    paths = config.paths.model_copy(
        update={
            "experiments_dir": Path("experiments/later"),
            "split_dir": Path("experiments/owner"),
        }
    )
    later = config.model_copy(update={"paths": paths})
    assert lock_path(later) == config.root / "experiments" / "owner" / "split_lock.json"
    assert manifest_path(later).parent == lock_path(later).parent
    with pytest.raises(ProjectError) as info:
        run_split(later)
    assert info.value.key == "split_owned_elsewhere"

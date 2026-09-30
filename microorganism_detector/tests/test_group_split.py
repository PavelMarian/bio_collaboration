"""Related-frame links and the non-temporal group split."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from sludge_micro.config import GroupingConfig
from sludge_micro.duplicates import UnionFind
from sludge_micro.group_split import SPLITS, assign_groups, group_vectors, split_cost
from sludge_micro.scenes import join_by, overlap_inliers, sampling_tokens, series_base


@pytest.mark.parametrize(
    ("name", "tokens"),
    [
        ("ч1.волокно..bmp", ("ch1",)),
        ("волокно.ч2..bmp", ("ch2",)),
        ("коловратки 2ч..bmp", ("ch2",)),
        ("зооглея.чаша2..bmp", ("ch2",)),
        ("Ил.Беспанцирная коловратка.jpg", ("sludge",)),
        ("раб.инф. 3 шт.ил.х100..bmp", ("sludge",)),
        ("коловратки. 1 а....bmp", ("aeration_1",)),
        ("а4..bmp", ("aeration_4",)),
        ("аэр.3..bmp", ("aeration_3", "number_3")),
        ("червь 1 аэр..bmp", ("aeration_1",)),
        ("3 отст 2 ст..bmp", ("settler",)),
        ("раб.колония 1..bmp", ("number_1",)),
        ("4.коловратка..bmp", ("number_4",)),
        ("цианобактерии..bmp", ()),
    ],
)
def test_sampling_marks_are_read_from_file_names(name, tokens):
    assert sampling_tokens(name) == tokens


def test_series_counter_is_removed():
    assert series_base("ч2.коловратка. (2)") == series_base("ч2.коловратка.")


def grouping():
    return GroupingConfig(
        orb_min_inliers=20,
        orb_features=1500,
        orb_ratio=0.75,
        orb_ransac_px=3.0,
        orb_image_width=640,
        filename_series=True,
        sampling_point_tokens=True,
        unmarked_per_session=True,
        search_iterations=10,
    )


def features(gray):
    orb = cv2.ORB_create(nfeatures=1500)
    points, descriptors = orb.detectAndCompute(gray, None)
    return np.float32([p.pt for p in points]), descriptors


def test_shifted_field_of_view_overlaps_and_other_scene_does_not():
    rng = np.random.default_rng(0)
    scene = cv2.GaussianBlur(rng.integers(0, 255, (600, 900), dtype=np.uint8), (5, 5), 0)
    first, shifted = scene[:480, :640], scene[60:540, 150:790]
    other = cv2.GaussianBlur(
        np.random.default_rng(1).integers(0, 255, (480, 640), dtype=np.uint8), (5, 5), 0
    )
    assert overlap_inliers(features(first), features(shifted), grouping(), 7) >= 20
    assert overlap_inliers(features(first), features(other), grouping(), 7) < 20


def test_shared_keys_join_frames():
    finder = UnionFind(4)
    join_by(finder, [[("point", "s", "ch1")], [("point", "s", "ch1")], [("point", "t", "ch1")], []])
    assert finder.find(0) == finder.find(1) != finder.find(2)
    assert finder.find(3) == 3


def synthetic_groups(count=12):
    rng = np.random.default_rng(3)
    rows = []
    for g in range(count):
        for i in range(int(rng.integers(1, 4))):
            rows.append({"image_id": f"g{g:02d}-{i}", "group": f"g{g:02d}"})
    table = pd.DataFrame(rows)
    anns = pd.DataFrame(
        [
            {
                "image_id": r.image_id,
                "supervision": "target",
                "class_name": "a" if k % 2 else "b",
                "x1": 0.0,
            }
            for k, r in enumerate(table.itertuples())
        ]
    )
    return table, anns


def test_assignment_keeps_groups_whole_and_repeats_for_one_seed(config):
    split = config.split.model_copy(update={"grouping": grouping()})
    config = config.model_copy(update={"split": split})
    table, anns = synthetic_groups()
    vectors = group_vectors(table, anns, ("a", "b"))
    first, cost = assign_groups(config, vectors)
    second, _ = assign_groups(config, vectors)
    assert first.equals(second) and set(first) <= set(SPLITS)
    per_image = table["group"].map(first)
    assert per_image.groupby(table["group"]).nunique().max() == 1
    assert cost >= 0


def test_cost_penalises_an_avoidable_missing_class():
    totals = np.array([10.0, 6.0])
    fractions = np.array([0.6, 0.2, 0.2])
    needs = np.array([False, True])
    balanced = np.array([[6.0, 4.0], [2.0, 1.0], [2.0, 1.0]])
    missing = np.array([[6.0, 6.0], [2.0, 0.0], [2.0, 0.0]])
    assert (
        split_cost(missing, totals, fractions, needs)
        > split_cost(balanced, totals, fractions, needs) + 10
    )

"""Matching and metrics on examples with known answers, plus metamorphic checks."""

from __future__ import annotations

import pandas as pd
import pytest

from sludge_micro.evaluate import (
    FP,
    IGNORED,
    TP,
    class_row,
    confidence_grid,
    match_all,
    match_image,
    select_confidence,
    summarise,
)

CLASSES = ("a", "b")


def frame(rows, score=False):
    cols = ["image_id", "class_name", "x1", "y1", "x2", "y2"] + (["score"] if score else [])
    return pd.DataFrame(rows, columns=cols)


def boxes(rows):
    return pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2"])


NO_IGNORE = boxes([])


def test_one_detection_matches_at_most_one_reference():
    det = frame([("i", "a", 0, 0, 10, 10, 0.9)], score=True)
    ref = frame([("i", "a", 0, 0, 10, 10), ("i", "a", 1, 1, 10, 10)])
    result = match_image(det, ref, NO_IGNORE, 0.5, 0.5)
    assert result.status == (TP,)
    assert result.matched_reference == (True, False)


def test_one_reference_gives_at_most_one_true_positive():
    det = frame([("i", "a", 0, 0, 10, 10, 0.9), ("i", "a", 0, 0, 10, 10, 0.8)], score=True)
    ref = frame([("i", "a", 0, 0, 10, 10)])
    assert match_image(det, ref, NO_IGNORE, 0.5, 0.5).status == (TP, FP)


def test_matching_requires_the_same_class():
    det = frame([("i", "b", 0, 0, 10, 10, 0.9)], score=True)
    ref = frame([("i", "a", 0, 0, 10, 10)])
    result = match_image(det, ref, NO_IGNORE, 0.5, 0.5)
    assert result.status == (FP,) and result.matched_reference == (False,)


def test_higher_confidence_claims_the_reference_first():
    det = frame([("i", "a", 0, 0, 10, 9, 0.6), ("i", "a", 0, 0, 10, 10, 0.9)], score=True)
    ref = frame([("i", "a", 0, 0, 10, 10)])
    assert match_image(det, ref, NO_IGNORE, 0.5, 0.5).status == (FP, TP)


def test_equal_confidence_resolves_by_detection_index_and_best_iou():
    det = frame([("i", "a", 0, 0, 10, 10, 0.5), ("i", "a", 0, 0, 10, 10, 0.5)], score=True)
    ref = frame([("i", "a", 0, 0, 10, 8), ("i", "a", 0, 0, 10, 10)])
    result = match_image(det, ref, NO_IGNORE, 0.5, 0.5)
    assert result.status == (TP, TP)
    assert result.matched_reference == (True, True)


def test_detection_inside_ignore_region_is_neither_tp_nor_fp():
    det = frame([("i", "a", 20, 20, 30, 30, 0.9), ("i", "a", 50, 50, 60, 60, 0.9)], score=True)
    ignore = boxes([("i", 18, 18, 32, 32)])
    assert match_image(det, frame([]), ignore, 0.5, 0.5).status == (IGNORED, FP)


def test_class_row_formula_and_undefined_values():
    row = class_row("a", tp=3, fp=1, support=6)
    assert row["precision"] == pytest.approx(0.75)
    assert row["recall"] == pytest.approx(0.5)
    assert row["f1"] == pytest.approx(2 * 3 / (2 * 3 + 1 + 3))
    empty = class_row("b", tp=0, fp=0, support=0)
    assert empty["f1"] is None and empty["precision"] is None and empty["recall"] is None
    assert class_row("c", tp=0, fp=2, support=0)["f1"] == 0.0


def evaluated():
    det = frame(
        [
            ("i1", "a", 0, 0, 10, 10, 0.9),
            ("i1", "a", 20, 20, 30, 30, 0.3),
            ("i2", "b", 0, 0, 10, 10, 0.2),
            ("i2", "a", 40, 40, 50, 50, 0.95),
        ],
        score=True,
    )
    ref = frame([("i1", "a", 0, 0, 10, 10), ("i2", "b", 0, 0, 10, 10), ("i2", "a", 5, 5, 9, 9)])
    return det, ref


def test_macro_f1_on_known_example(config):
    det, ref = evaluated()
    det_m, ref_m = match_all(det, ref, NO_IGNORE, config.evaluation)
    result = summarise(det_m, ref_m, CLASSES, 0.25)
    by_name = {r["name"]: r for r in result["classes"]}
    assert (by_name["a"]["tp"], by_name["a"]["fp"], by_name["a"]["fn"]) == (1, 2, 1)
    assert (by_name["b"]["tp"], by_name["b"]["fp"], by_name["b"]["fn"]) == (0, 0, 1)
    assert result["macro_f1"] == pytest.approx((2 / 5 + 0.0) / 2)


def test_class_without_support_and_predictions_is_excluded(config):
    det, ref = evaluated()
    det_m, ref_m = match_all(det, ref, NO_IGNORE, config.evaluation)
    result = summarise(det_m, ref_m, (*CLASSES, "c"), 0.25)
    assert result["macro_f1_classes"] == ["a", "b"]


def test_counts_do_not_depend_on_detection_order(config):
    det, ref = evaluated()
    shuffled = det.sample(frac=1.0, random_state=3).reset_index(drop=True)
    first = summarise(*match_all(det, ref, NO_IGNORE, config.evaluation), CLASSES, 0.25)
    second = summarise(*match_all(shuffled, ref, NO_IGNORE, config.evaluation), CLASSES, 0.25)
    assert first == second


def test_threshold_selection_prefers_best_then_higher(config):
    det, ref = evaluated()
    det_m, ref_m = match_all(det, ref, NO_IGNORE, config.evaluation)
    chosen, curve = select_confidence(det_m, ref_m, CLASSES, config.evaluation)
    best = max(c["macro_f1"] for c in curve if c["macro_f1"] is not None)
    ties = [c["confidence"] for c in curve if c["macro_f1"] == best]
    assert chosen == max(ties)
    assert [c["confidence"] for c in curve] == confidence_grid(config.evaluation)


def test_count_errors_follow_the_ignore_rule_and_include_empty_frames(config):
    from sludge_micro.evaluate import count_errors

    det = frame(
        [
            ("i", "a", 0, 0, 10, 10, 0.9),
            ("i", "a", 50, 50, 60, 60, 0.8),
            ("i", "b", 0, 0, 10, 10, 0.1),
            ("j", "a", 20, 20, 30, 30, 0.7),
        ],
        score=True,
    )
    ref = frame([("i", "a", 0, 0, 10, 10), ("i", "b", 0, 0, 10, 10), ("k", "b", 5, 5, 9, 9)])
    ignore = boxes([("j", 18, 18, 32, 32)])
    matched, refs = match_all(det, ref, ignore, config.evaluation)
    result = count_errors(matched, refs, ["i", "j", "k", "empty"], CLASSES, 0.5)
    rows = {r["name"]: r for r in result["classes"]}
    assert result["images"] == 4
    assert rows["a"] == {
        "name": "a",
        "annotated": 1,
        "counted": 2,
        "in_ignore_regions": 1,
        "mae_per_frame": 0.25,
        "bias_per_frame": 0.25,
        "mae_per_frame_all": 0.5,
        "bias_per_frame_all": 0.5,
    }
    assert rows["b"]["counted"] == 0 and rows["b"]["annotated"] == 2
    assert rows["b"]["mae_per_frame"] == 0.5 and rows["b"]["bias_per_frame"] == -0.5
    shuffled, refs2 = match_all(det.iloc[::-1], ref, ignore, config.evaluation)
    assert count_errors(shuffled, refs2, ["i", "j", "k", "empty"], CLASSES, 0.5) == result

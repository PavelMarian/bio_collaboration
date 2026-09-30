"""Committed passport of the delivered model and the files derived from it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sludge_micro.adapter import categories_document, pavel_block
from sludge_micro.config import load_config
from sludge_micro.delivery import render_summary
from sludge_micro.messages import ProjectError
from sludge_micro.passport import ModelPassport, check_weights, load_passport

ROOT = Path(__file__).resolve().parents[1]
PASSPORT = ROOT / "model" / "passport.json"
WEIGHTS = ROOT / "weights" / "microorganisms_fasterrcnn.pt"


@pytest.fixture(scope="module")
def passport() -> ModelPassport:
    return load_passport(PASSPORT)


def test_passport_matches_the_working_configuration(passport, config):
    assert passport.names == config.classes.names
    assert [c.title for c in passport.classes] == [c.title for c in config.classes.target]
    assert passport.image.min_size == config.model.min_size
    assert passport.image.max_size == config.model.max_size
    assert passport.image.formats == config.audit.image_suffixes
    post = config.postprocess
    d = passport.detector
    assert (d.min_score, d.nms_iou, d.max_detections) == (
        post.min_score,
        post.nms_iou,
        post.max_detections,
    )
    assert d.architecture == config.model.architecture
    assert passport.evaluation["validation"]["confidence"] == d.confidence


def test_training_recipe_repeats_the_settings_of_the_delivered_model(passport):
    recipe = load_config(ROOT / "configs" / "frcnn_bn_frozen.yaml")
    t, trained = recipe.train, passport.training
    assert t.epochs == trained["epochs"]
    assert (t.lr, t.momentum, t.weight_decay) == (
        trained["optimizer"]["lr"],
        trained["optimizer"]["momentum"],
        trained["optimizer"]["weight_decay"],
    )
    assert (t.batch_size, t.accumulate, t.batchnorm, t.sampling) == (
        trained["batch_size"],
        trained["accumulate"],
        trained["batchnorm"],
        trained["sampling"],
    )
    assert t.augment.model_dump(mode="json") == trained["augment"]
    assert recipe.model.trainable_backbone_layers == trained["trainable_backbone_layers"]
    assert (recipe.runtime.seed, recipe.runtime.device) == (trained["seed"], trained["device"])
    assert recipe.prepare.variant == trained["data_variant"]
    assert recipe.split.protocol == trained["split"]["protocol"]
    assert Path(recipe.paths.pretrained_weights).name == trained["initial_weights"]["file"]


def test_readme_block_is_rendered_from_the_passport(passport):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start, end = "<!-- generated:passport -->", "<!-- /generated:passport -->"
    assert f"{start}\n\n{render_summary(passport)}\n\n{end}" in readme


def test_pipeline_files_are_the_export_of_the_passport(passport):
    categories = categories_document([(c.label, c.name) for c in passport.classes])
    d = passport.detector
    block = pavel_block(d.architecture, d.confidence, d.max_detections, passport.names)
    folder = ROOT / "model" / "pavel"
    assert (folder / "microorganism_instances.json").read_text(encoding="utf-8") == json.dumps(
        categories, indent=2
    )
    assert (folder / "microorganism_block.yaml").read_text(encoding="utf-8") == yaml.safe_dump(
        block, allow_unicode=True
    )


def test_test_result_is_marked_as_single_use(passport):
    test = passport.evaluation["test"]
    assert test["single_use"] is True and test["confidence"] == passport.detector.confidence
    assert len(test["counting"]) == len(passport.classes)


def test_passport_rejects_unordered_labels(tmp_path):
    data = json.loads(PASSPORT.read_text(encoding="utf-8"))
    data["classes"][0]["label"] = 2
    path = tmp_path / "passport.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProjectError) as info:
        load_passport(path)
    assert info.value.key == "passport_invalid"


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="weights are downloaded from the release")
def test_downloaded_weights_match_the_passport(passport):
    check_weights(passport, WEIGHTS)

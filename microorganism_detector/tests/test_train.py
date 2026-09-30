"""End-to-end pipeline on synthetic data: training determinism, single test use,
budget and one-command reproduction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from sludge_micro.audit import run_audit
from sludge_micro.config import load_config
from sludge_micro.messages import ProjectError
from sludge_micro.model import build_detector_raw
from sludge_micro.prepare import run_prepare
from sludge_micro.runs import evaluate_run
from sludge_micro.split import run_split
from sludge_micro.train import run_train

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def pretrained(tmp_path_factory):
    path = tmp_path_factory.mktemp("weights") / "coco.pth"
    torch.manual_seed(1)
    torch.save(build_detector_raw("fasterrcnn_resnet50_fpn_v2", 91, None, None).state_dict(), path)
    return path


def build_archive(helpers, config, raw, count):
    rot, att = 3, 2
    images, anns, files = [], [], {}
    for i in range(1, count + 1):
        name = f"t/{i}.bmp"
        images.append({"id": i, "file_name": name, "width": 96, "height": 72, "date_captured": 0})
        anns.append(helpers.box_ann(2 * i - 1, i, rot, [10 + i, 10, 30, 25]))
        anns.append(helpers.box_ann(2 * i, i, att, [50, 30 + i, 30, 30]))
        if i == 2:
            anns.append(helpers.box_ann(100 + i, i, 15, [5, 50, 10, 10]))
        files[name] = helpers.encode(helpers.synthetic_image(100 + i, 96, 72), ".bmp")
    helpers.write_coco_zip(
        raw / "T.zip", helpers.coco_document(config, images, anns), files, "Train"
    )


@pytest.fixture
def trainable(tmp_path, config, helpers, pretrained):
    root = tmp_path / "proj"
    for folder in ("configs", "experiments", "data/raw"):
        (root / folder).mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'synthetic'\n")
    build_archive(helpers, config, root / "data" / "raw", 9)
    months = ["01"] * 3 + ["02"] * 3 + ["03"] * 3
    owner = pd.DataFrame(
        {
            "image_id": [f"arch_t-{i:04d}" for i in range(1, 10)],
            "captured_at": [f"2024-{m}-0{1 + i % 3}T10:00:00" for i, m in enumerate(months)],
        }
    )
    owner.to_csv(root / "owner.csv", index=False)
    data = helpers.project_config(config)
    data["sources"] = [
        {
            "id": "arch_t",
            "archive": "T.zip",
            "format": "coco",
            "role": "microorganisms",
            "priority": 0,
        }
    ]
    data["audit"].update(expected_width=96, expected_height=72)
    data["duplicate_variants"] = {"exclude_conflicts": None}
    data["paths"]["pretrained_weights"] = str(pretrained)
    data["model"].update(min_size=72, max_size=96)
    data["train"].update(epochs=1)
    data["split"].update(
        owner_metadata="owner.csv",
        train_end="2024-01-31T23:59:59",
        validation_end="2024-02-29T23:59:59",
    )
    path = root / "configs" / "microorganisms.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True))
    config = load_config(path)
    run_audit(config)
    run_prepare(config)
    run_split(config)
    return config


def state(config, run_id):
    return torch.load(config.root / "outputs" / run_id / "model.pt", weights_only=False)["model"]


def test_training_is_deterministic_and_reports_validation(trainable):
    first = run_train(trainable, "control")
    second = run_train(trainable, "repeat")
    assert {k: v for k, v in first.items() if k != "run_id"} == {
        k: v for k, v in second.items() if k != "run_id"
    }
    a, b = state(trainable, "control"), state(trainable, "repeat")
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert first["split"] == "validation" and first["images"] == 3
    assert first["confidence_source"] == "validation_selection"
    assert first["period"] == ["2024-02-01T10:00:00", "2024-02-03T10:00:00"]
    manifest = (trainable.root / "experiments" / "control" / "manifest.json").read_text()
    assert "split_lock" in manifest and "config_sha256" in manifest
    import json

    sampling = json.loads(manifest)["sampling"]
    assert (
        sampling["policy"] == "none" and sampling["images"] == json.loads(manifest)["train_images"]
    )
    assert set(sampling["presence"]) == set(trainable.classes.names)


def test_test_split_is_used_once_and_budget_is_enforced(trainable):
    import json

    run_train(trainable, "control")
    record = evaluate_run(trainable, "control", "test")
    assert record["confidence_source"] == "frozen_from_validation"
    assert record["counting"]["images"] == record["images"]
    assert sum(c["annotated"] for c in record["counting"]["classes"]) == record["objects"]
    usage = json.loads((trainable.root / "experiments" / "test_usage.json").read_text())
    assert usage["run_id"] == "control" and usage["confidence"] == record["confidence"]
    assert all(v for k, v in usage["run_check"].items() if not k.endswith("sha256"))
    assert usage["code"]["package_sha256"] and usage["predictions"].endswith("test.parquet")
    with pytest.raises(ProjectError) as info:
        evaluate_run(trainable, "control", "test")
    assert info.value.key == "test_already_used"
    limited = trainable.model_copy(
        update={"protocol": trainable.protocol.model_copy(update={"max_runs": 1})}
    )
    with pytest.raises(ProjectError) as budget:
        run_train(limited, "extra")
    assert budget.value.key == "budget_exhausted"


def test_candidate_hypotheses_train_on_the_same_split(trainable):
    root = trainable.root / "configs" / "experiments"
    root.mkdir()
    (root / "rotation90.yaml").write_text(
        "extends: ../microorganisms.yaml\ntrain:\n  augment: {rotation90: true}\n"
    )
    (root / "rarest_sampling.yaml").write_text(
        "extends: ../microorganisms.yaml\ntrain:\n  sampling: rarest_class_presence\n"
    )
    control = run_train(trainable, "control")
    for name in ("rotation90", "rarest_sampling"):
        candidate = load_config(root / f"{name}.yaml")
        record = run_train(candidate, name)
        assert record["images"] == control["images"] and record["objects"] == control["objects"]
        assert record["period"] == control["period"]


def test_predict_and_export_commands_are_repeatable(trainable, helpers):
    import cv2

    from sludge_micro.cli import main

    run_train(trainable, "control")
    images = trainable.root / "new_images"
    images.mkdir()
    for i in range(3):
        cv2.imwrite(str(images / f"{i}.bmp"), helpers.synthetic_image(300 + i, 96, 72))
    config_path = str(trainable.root / "configs" / "microorganisms.yaml")
    outputs = []
    for name in ("a", "b"):
        target = trainable.root / "outputs" / f"predict_{name}"
        args = [
            "predict",
            "--config",
            config_path,
            "--run-id",
            "control",
            "--input",
            str(images),
            "--output",
            str(target),
        ]
        assert main(args) == 0
        outputs.append(target)
    for file in ("microorganism_detections.json", "microorganism_counts.csv"):
        assert (outputs[0] / file).read_bytes() == (outputs[1] / file).read_bytes()
    export = trainable.root / "outputs" / "export"
    assert (
        main(
            [
                "export-pavel",
                "--config",
                config_path,
                "--run-id",
                "control",
                "--output",
                str(export),
            ]
        )
        == 0
    )
    assert (export / "annotations" / "microorganism_instances.json").is_file()


def test_test_split_requires_a_validated_run(trainable):
    with pytest.raises(ProjectError) as info:
        evaluate_run(trainable, "missing", "test")
    assert info.value.key == "confidence_unselected"


def candidate(trainable, name, text):
    folder = trainable.root / "configs" / "experiments"
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.yaml").write_text("extends: ../microorganisms.yaml\n" + text)
    return load_config(folder / f"{name}.yaml")


def test_retinanet_candidate_trains_deterministically(trainable):
    coco = trainable.root / "outputs" / "pretrained" / "retina_coco.pth"
    coco.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2)
    torch.save(build_detector_raw("retinanet_resnet50_fpn_v2", 91, None, None).state_dict(), coco)
    config = candidate(
        trainable,
        "retinanet",
        (
            "model: {architecture: retinanet_resnet50_fpn_v2}\n"
            f"paths: {{pretrained_weights: {coco}}}\n"
        ),
    )
    first = run_train(config, "retina_a")
    second = run_train(config, "retina_b")
    assert {k: v for k, v in first.items() if k != "run_id"} == {
        k: v for k, v in second.items() if k != "run_id"
    }
    a, b = state(config, "retina_a"), state(config, "retina_b")
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_yolo_candidate_trains_through_the_same_protocol(trainable, fake_ultralytics, helpers):
    names = list(trainable.classes.names)
    weights = helpers.fake_weights(
        trainable.root / "outputs" / "pretrained" / "yolo.pt",
        names,
        [[10.0, 10.0, 40.0, 35.0, 0.8, 2.0]],
    )
    config = candidate(
        trainable,
        "yolo",
        (
            f"model:\n  architecture: ultralytics_yolo\n"
            f"  yolo: {{weights: {weights}, image_size: 96, ignore_policy: exclude_image}}\n"
        ),
    )
    record = run_train(config, "yolo")
    train_call = next(kwargs for kind, kwargs in fake_ultralytics.calls if kind == "train")
    assert train_call["seed"] == config.runtime.seed and train_call["deterministic"] is True
    data = yaml.safe_load(Path(train_call["data"]).read_text(encoding="utf-8"))
    assert list(data["names"].values()) == names
    assert record["split"] == "validation" and record["images"] == 3
    assert (config.root / "outputs" / "yolo" / "model.pt").is_file()


def test_training_filter_leaves_out_ignore_images_from_train_only(trainable):
    from sludge_micro.runs import split_ids

    base = split_ids(trainable, "train")
    filtered_config = trainable.model_copy(
        update={"train": trainable.train.model_copy(update={"exclude_ignore_images": True})}
    )
    assert split_ids(filtered_config, "train") == base - {"arch_t-0002"}
    assert split_ids(filtered_config, "validation") == split_ids(trainable, "validation")


def test_group_protocol_splits_whole_groups_and_locks_statistics(trainable):
    from sludge_micro.config import GroupingConfig
    from sludge_micro.split import load_split

    grouping = GroupingConfig(
        orb_min_inliers=20,
        orb_features=500,
        orb_ratio=0.75,
        orb_ransac_px=3.0,
        orb_image_width=96,
        filename_series=False,
        sampling_point_tokens=False,
        unmarked_per_session=False,
        search_iterations=20,
    )
    split = trainable.split.model_copy(
        update={"protocol": "group", "grouping": grouping, "alternative_approval": "test approval"}
    )
    config = trainable.model_copy(update={"split": split})
    lock = run_split(config)
    manifest = load_split(config)
    assert manifest.groupby("group")["split"].nunique().max() == 1
    assert {s: v["images"] for s, v in lock["splits"].items()} == manifest[
        "split"
    ].value_counts().to_dict()
    assert all(v["images"] > 0 for v in lock["splits"].values())
    assert lock["train_filter"]["enabled"] is False and manifest["captured_at"].isna().all()


def test_comparison_examples_and_pipeline_handover_on_a_trained_run(trainable):
    from sludge_micro.compare import run_comparison
    from sludge_micro.examples import run_examples
    from sludge_micro.handover import run_handover

    run_train(trainable, "control")
    # The hand-over copies cv_module from the pipeline folder: the root of the shared repository.
    pipeline = Path(__file__).resolve().parents[2]
    paths = trainable.paths.model_copy(update={"pavel_pipeline": pipeline})
    trainable = trainable.model_copy(update={"paths": paths})
    comparison = run_comparison(trainable)
    control_f1 = comparison["runs"][0]["macro_f1"]
    assert comparison["decision"] == {
        "control": "control",
        "control_macro_f1": control_f1,
        "failed_checks": {},
        "accepted": [],
        "selected": "control",
        "rule": "candidate_greater_than_control",
    }
    assert comparison["same_validation"] is True
    assert comparison["same_split"] is True and comparison["same_train"] is True
    control = comparison["runs"][0]
    assert control["trainable_parameters"] == control["trainable_parameters_by_config"]
    assert control["train"]["batchnorm"] == "batch" and control["batchnorm_statistics_changed"] > 0
    assert control["train_images"] == comparison["train_images_expected"]
    assert control["train_ids_match"] is None and control["code"]["package_sha256"]
    report = run_examples(trainable, "control", 2)
    kinds = {e["kind"] for e in report["examples"]}
    assert kinds <= {"correct", "false_positive", "missed"} and report["examples"]
    assert all((trainable.root / e["file"]).is_file() for e in report["examples"])
    handover = run_handover(trainable, "control", 3)
    assert handover["class_errors"] == []
    assert handover["export_weights_identical"] is True
    assert handover["images"] == 3
    assert handover["detections_match"] == 3 and handover["counts_match_detections"] == 3
    assert handover["count_columns"] == ["image_id", *trainable.classes.names]


def test_resume_continues_exactly_and_runs_are_not_overwritten(trainable):
    two = trainable.model_copy(update={"train": trainable.train.model_copy(update={"epochs": 2})})
    run_train(two, "straight")
    run_train(trainable, "resumed")
    run_train(two, "resumed", resume=True)
    a, b = state(two, "straight"), state(two, "resumed")
    assert all(torch.equal(a[k], b[k]) for k in a)
    manifest = (two.root / "experiments" / "resumed" / "manifest.json").read_text()
    assert '"resumed_after_epoch": 1' in manifest
    with pytest.raises(ProjectError) as info:
        run_train(two, "straight")
    assert info.value.key == "run_exists"
    with pytest.raises(ProjectError) as missing:
        run_train(two, "absent", resume=True)
    assert missing.value.key == "resume_missing"


def test_frozen_batchnorm_keeps_stored_statistics_while_training(config):
    from sludge_micro.model import build_detector, freeze_backbone
    from sludge_micro.train import (
        batchnorm_check,
        batchnorm_statistics,
        train_epoch,
        trainable_parameters,
    )

    small = config.model_copy(
        update={"model": config.model.model_copy(update={"min_size": 96, "max_size": 128})}
    )
    image = torch.rand(3, 48, 64, generator=torch.Generator().manual_seed(0))
    target = {"boxes": torch.tensor([[4.0, 4.0, 40.0, 30.0]]), "labels": torch.tensor([1])}
    changed, checks, trainable = {}, {}, {}
    for mode in ("batch", "frozen"):
        torch.manual_seed(0)
        model = build_detector(small)
        freeze_backbone(model, small.model.trainable_backbone_layers)
        stored = {k: v.clone() for k, v in model.state_dict().items() if "running_" in k}
        start = batchnorm_statistics(model)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        train_epoch(model, [((image,), (target,))], optimizer, 1, torch.device("cpu"), mode)
        after = model.state_dict()
        changed[mode] = sum(not torch.equal(stored[k], after[k]) for k in stored)
        checks[mode] = batchnorm_check(model, start, mode)
        trainable[mode] = trainable_parameters(model)
    assert changed["batch"] > 0
    assert changed["frozen"] == 0
    assert checks["frozen"]["changed"] == 0 and checks["frozen"]["layers_in_training_mode"] == 0
    assert checks["batch"]["changed"] > 0
    assert checks["batch"]["layers_in_training_mode"] == checks["batch"]["layers"] > 0
    assert trainable["batch"] == trainable["frozen"]


def comparison_record(run_id: str, macro_f1: float, **changes: object) -> dict:
    record = {
        "run_id": run_id,
        "architecture": "fasterrcnn_resnet50_fpn_v2",
        "split": "validation",
        "images": 46,
        "objects": 172,
        "ignore_regions": 8,
        "macro_f1": macro_f1,
        "split_manifest_sha256": "lock",
        "train_images": 103,
        "train_ids_match": None,
        "train": {"batchnorm": "batch"},
        "batchnorm_statistics_changed": None,
    }
    return record | changes


def test_comparison_uses_the_named_control_and_mandatory_checks(config):
    from sludge_micro.compare import decide

    frozen = {"batchnorm": "frozen"}
    records = [
        comparison_record("control", 0.2380),
        comparison_record("bn", 0.2381, train=frozen, batchnorm_statistics_changed=0),
        comparison_record("bn_changed", 0.30, train=frozen, batchnorm_statistics_changed=3),
        comparison_record("other_split", 0.30, split_manifest_sha256="other"),
        comparison_record("other_validation", 0.30, objects=171),
        comparison_record("equal", 0.2380),
    ]
    protocol = config.protocol.model_copy(update={"control_run": "control"})
    named = config.model_copy(update={"protocol": protocol})
    decision = decide(named, records, "lock", 103)
    assert decision["control"] == "control" and decision["control_macro_f1"] == 0.2380
    assert decision["accepted"] == ["bn"] and decision["selected"] == "bn"
    assert decision["failed_checks"] == {
        "bn_changed": ["batchnorm_statistics"],
        "other_split": ["split"],
        "other_validation": ["validation"],
    }
    assert decide(config, records, "lock", 103)["control"] is None


def test_test_evaluation_refuses_a_run_that_differs_from_its_manifest(trainable):
    import json

    run_train(trainable, "control")
    path = trainable.root / "experiments" / "control" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["inputs"]["checkpoint"]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ProjectError) as info:
        evaluate_run(trainable, "control", "test")
    assert info.value.key == "test_run_mismatch"
    assert "checkpoint_matches_manifest" in str(info.value)
    assert not (trainable.root / "experiments" / "test_usage.json").exists()
    assert not (trainable.root / "outputs" / "control" / "predictions_test.parquet").exists()


def test_comparison_takes_the_control_from_an_earlier_stage(config, tmp_path):
    import json

    from sludge_micro.compare import metrics_files

    for folder in ("stage2/a", "stage2/b", "stage1/control", "stage1/other"):
        (tmp_path / "experiments" / folder).mkdir(parents=True)
        (tmp_path / "experiments" / folder / "metrics.json").write_text(json.dumps({}))
    paths = config.paths.model_copy(
        update={
            "experiments_dir": Path("experiments/stage2"),
            "split_dir": Path("experiments/stage1"),
        }
    )
    protocol = config.protocol.model_copy(
        update={"control_run": "control", "control_dir": Path("experiments/stage1")}
    )
    prepare = config.prepare.model_copy(update={"variant": "prefer_student_6"})
    later = config.model_copy(
        update={"root": tmp_path, "paths": paths, "protocol": protocol, "prepare": prepare}
    )
    names = [
        p.parent.relative_to(tmp_path / "experiments").as_posix() for p in metrics_files(later)
    ]
    assert names == ["stage1/control", "stage2/a", "stage2/b"]
    missing = later.model_copy(
        update={"protocol": protocol.model_copy(update={"control_run": "absent"})}
    )
    with pytest.raises(ProjectError) as info:
        metrics_files(missing)
    assert info.value.key == "run_missing"


def test_passport_of_a_trained_run_drives_predict_export_and_checks(
    trainable, helpers, monkeypatch
):
    import json

    import cv2

    from sludge_micro.cli import main

    run_train(trainable, "control")
    evaluate_run(trainable, "control", "test")
    root = trainable.root
    config_path = str(root / "configs" / "microorganisms.yaml")
    passport_path = root / "delivery" / "passport.json"
    passport_path.parent.mkdir()
    args = ["--config", config_path, "--run-id", "control"]
    assert main(["passport", *args, "--output", str(passport_path)]) == 0
    record = json.loads(passport_path.read_text(encoding="utf-8"))
    weights = root / "outputs" / "control" / "model.pt"
    metrics = json.loads((root / "experiments" / "control" / "metrics.json").read_text())
    assert record["detector"]["confidence"] == metrics["confidence"]
    assert record["evaluation"]["validation"]["macro_f1"] == metrics["macro_f1"]
    assert record["evaluation"]["test"]["single_use"] is True
    images = root / "new_images"
    images.mkdir()
    for i in range(3):
        cv2.imwrite(str(images / f"{i}.bmp"), helpers.synthetic_image(400 + i, 96, 72))
    # Passport commands run where no project configuration exists.
    monkeypatch.chdir(passport_path.parent)
    delivered = ["--passport", str(passport_path), "--weights", str(weights)]
    by_run, by_passport = root / "out_run", root / "out_passport"
    assert main(["predict", *args, "--input", str(images), "--output", str(by_run)]) == 0
    assert main(["predict", *delivered, "--input", str(images), "--output", str(by_passport)]) == 0
    for name in ("microorganism_detections.json", "microorganism_counts.csv"):
        assert (by_run / name).read_bytes() == (by_passport / name).read_bytes()
    assert main(["check-model", *delivered]) == 0
    export_run, export_passport = root / "export_run", root / "export_passport"
    assert main(["export-pavel", *args, "--output", str(export_run)]) == 0
    assert main(["export-pavel", *delivered, "--output", str(export_passport)]) == 0
    for name in (
        "microorganism_block.yaml",
        "annotations/microorganism_instances.json",
        "models/microorganisms_fasterrcnn.pt",
    ):
        assert (export_run / name).read_bytes() == (export_passport / name).read_bytes()
    check_delivery_refusals(root, record, passport_path, weights, images)


def check_delivery_refusals(root, record, passport_path, weights, images):
    from sludge_micro.cli import main
    from sludge_micro.passport import ModelPassport, passport_model

    tampered = root / "tampered.pt"
    tampered.write_bytes(weights.read_bytes() + b"x")
    assert main(["check-model", "--passport", str(passport_path), "--weights", str(tampered)]) == 2
    first, second = record["classes"][:2]
    swapped = record | {
        "classes": [second | {"label": 1}, first | {"label": 2}, *record["classes"][2:]]
    }
    with pytest.raises(ProjectError) as info:
        passport_model(ModelPassport.model_validate(swapped), weights)
    assert info.value.key == "passport_checkpoint_mismatch"
    no_weights = ["predict", "--passport", str(passport_path), "--input", str(images)]
    assert main([*no_weights, "--output", str(root / "none")]) == 2
    readme = root / "README.md"
    readme.write_text("# R\n\n<!-- generated:passport -->\n\n<!-- /generated:passport -->\n")
    assert main(["passport-readme", "--passport", str(passport_path), "--readme", str(readme)]) == 0
    assert "`control`" in readme.read_text(encoding="utf-8")
    report = root / "pipeline_check.json"
    pipeline = Path(__file__).resolve().parents[2]
    delivered = ["--passport", str(passport_path), "--weights", str(weights)]
    check = ["--input", str(images), "--pipeline", str(pipeline), "--report", str(report)]
    assert main(["check-pipeline", *delivered, *check]) == 0
    import json

    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["detections_match"] == result["images"] == result["counts_match_detections"] == 3
    assert result["class_errors"] == [] and result["export_weights_identical"] is True

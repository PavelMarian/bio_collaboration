"""RetinaNet v2 candidate: loading, inference contract, ignore regions, export."""

from __future__ import annotations

import pytest
import torch

from sludge_micro.checkpoint import architecture_hint, head_outputs
from sludge_micro.config import load_config
from sludge_micro.infer import Predictor
from sludge_micro.messages import ProjectError
from sludge_micro.model import (
    build_detector,
    build_detector_raw,
    init_from_pretrained,
    load_trained,
    save_checkpoint,
)
from sludge_micro.prepare import IGNORE_LABEL
from sludge_micro.retinanet import BACKGROUND_LOGIT, IgnoreAwareRetinaNet, background_channels

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def retina_config():
    config = load_config(
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "configs"
        / "microorganisms.yaml"
    )
    model = config.model.model_copy(
        update={"architecture": "retinanet_resnet50_fpn_v2", "min_size": 72, "max_size": 96}
    )
    return config.model_copy(update={"model": model})


def forced(model):
    """Raise every target class score so that the untrained model emits detections."""
    conv = model.head.classification_head.cls_logits
    keep = set(background_channels(model).tolist())
    with torch.no_grad():
        for channel in range(conv.bias.numel()):
            if channel not in keep:
                conv.bias[channel] = 0.0
    return model


def test_candidate_builds_ignore_aware_model_with_known_head(retina_config):
    torch.manual_seed(0)
    model = build_detector(retina_config)
    state = model.state_dict()
    assert isinstance(model, IgnoreAwareRetinaNet)
    assert head_outputs(state) == len(retina_config.classes.names) + 1
    assert architecture_hint(state) == "retinanet_resnet50_fpn_v2"


def test_prediction_contract_never_returns_background(retina_config, helpers):
    torch.manual_seed(0)
    model = forced(build_detector(retina_config))
    image = helpers.synthetic_image(4, 96, 72)
    copy = image.copy()
    objects = Predictor(model, retina_config.classes.names, 0.05, torch.device("cpu")).predict_bgr(
        image
    )
    assert objects and (image == copy).all()
    assert {o.class_name for o in objects} <= set(retina_config.classes.names)
    for o in objects:
        x1, y1, x2, y2 = o.bbox_xyxy
        assert 0 <= x1 < x2 <= 96 and 0 <= y1 < y2 <= 72 and 0 <= o.confidence <= 1


def test_anchors_on_ignore_regions_leave_both_losses(retina_config):
    torch.manual_seed(0)
    model = build_detector(retina_config).train()
    captured = []
    original = model.head.compute_loss

    def spy(targets, head_outputs, anchors, matched):
        captured.extend(matched)
        return original(targets, head_outputs, anchors, matched)

    model.head.compute_loss = spy
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 60.0, 50.0]]),
        "labels": torch.tensor([IGNORE_LABEL]),
    }
    losses = model([torch.rand(3, 72, 96)], [target])
    assert all(torch.isfinite(v) for v in losses.values())
    assert int((captured[0] >= 0).sum()) == 0
    assert int((captured[0] == model.proposal_matcher.BETWEEN_THRESHOLDS).sum()) > 0


def test_background_column_stays_frozen_during_training(retina_config):
    torch.manual_seed(0)
    model = build_detector(retina_config).train()
    conv = model.head.classification_head.cls_logits
    channels = background_channels(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=5e-4)
    target = {"boxes": torch.tensor([[5.0, 5.0, 40.0, 40.0]]), "labels": torch.tensor([3])}
    for _ in range(3):
        loss = sum(model([torch.rand(3, 72, 96)], [target]).values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(conv.weight.detach()[channels].abs().max()) == 0.0
    assert float(conv.bias.detach()[channels].max()) < BACKGROUND_LOGIT + 0.01


def test_coco_weights_load_except_class_logits(retina_config, tmp_path):
    torch.manual_seed(1)
    coco = build_detector_raw("retinanet_resnet50_fpn_v2", 91, None, None)
    path = tmp_path / "retina_coco.pth"
    torch.save(coco.state_dict(), path)
    model = build_detector(retina_config)
    info = init_from_pretrained(model, path, "retinanet_resnet50_fpn_v2")
    assert info["skipped"] == ["head.classification_head.cls_logits"]
    name = "backbone.body.layer1.0.conv1.weight"
    assert torch.equal(model.state_dict()[name], coco.state_dict()[name])
    conv = model.head.classification_head.cls_logits
    assert float(conv.weight.detach()[background_channels(model)].abs().max()) == 0.0


def test_checkpoint_round_trip_and_architecture_check(retina_config, config, tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "retina.pt"
    save_checkpoint(build_detector(retina_config), path, retina_config, {"run_id": "r"})
    model, meta = load_trained(path, retina_config)
    assert isinstance(model, IgnoreAwareRetinaNet) and meta["config"]["architecture"] == (
        "retinanet_resnet50_fpn_v2"
    )
    with pytest.raises(ProjectError) as info:
        load_trained(path, config)
    assert info.value.key == "checkpoint_architecture_mismatch"

"""Detector construction, initialisation and checkpoint format.

Own checkpoints are dictionaries with ``model`` (state dict), ``config``
(``architecture``, ``min_size``, ``max_size``) and ``classes`` (names in label
order, label ``i`` is ``classes[i - 1]``). The Pavel torchvision backend reads
``model`` and ``config`` from the same layout.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

from sludge_micro.archives import sha256_file
from sludge_micro.checkpoint import head_outputs, load_restricted, state_dict_of
from sludge_micro.config import ProjectConfig
from sludge_micro.messages import ProjectError
from sludge_micro.retinanet import build_retinanet

LOG = logging.getLogger(__name__)
CHECKPOINT_FORMAT = "sludge_micro/1"
BUILDERS = {
    "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
    "retinanet_resnet50_fpn_v2": build_retinanet,
}
# Builder argument names of the shared post-processing settings.
POSTPROCESS_KEYS = {
    "fasterrcnn_resnet50_fpn_v2": ("box_score_thresh", "box_nms_thresh", "box_detections_per_img"),
    "retinanet_resnet50_fpn_v2": ("score_thresh", "nms_thresh", "detections_per_img"),
}
# Class-specific parameters that COCO weights cannot initialise.
CLASS_SPECIFIC = {
    "fasterrcnn_resnet50_fpn_v2": ("roi_heads.box_predictor.", "roi_heads.mask_"),
    "retinanet_resnet50_fpn_v2": ("head.classification_head.cls_logits.",),
}
Detector = GeneralizedRCNN | nn.Module
BACKBONE_LAYERS = ("layer4", "layer3", "layer2", "layer1", "conv1")
HASH_SUFFIX = re.compile(r"-([0-9a-f]{8,64})\.pth$")


def build_detector_raw(
    architecture: str, outputs: int, min_size: int | None, max_size: int | None, **extra: float
) -> Detector:
    """Build an untrained detector; no weights are downloaded.

    Args:
        architecture: torchvision builder name.
        outputs: Classifier outputs including background.
        min_size: Resize short side, or ``None`` for the torchvision default.
        max_size: Resize long-side bound, or ``None`` for the torchvision default.
        **extra: Post-processing arguments passed to the builder.
    """
    sizes = {k: v for k, v in (("min_size", min_size), ("max_size", max_size)) if v is not None}
    builder = BUILDERS[architecture]
    return builder(weights=None, weights_backbone=None, num_classes=outputs, **sizes, **extra)


def build_detector(config: ProjectConfig) -> Detector:
    """Build the configured torchvision detector with configured post-processing."""
    architecture = config.model.architecture
    post = config.postprocess
    values = (post.min_score, post.nms_iou, post.max_detections)
    extra = dict(zip(POSTPROCESS_KEYS[architecture], values, strict=True))
    return build_detector_raw(
        architecture,
        len(config.classes.target) + 1,
        config.model.min_size,
        config.model.max_size,
        **extra,
    )


def freeze_backbone(model: Detector, trainable_layers: int) -> None:
    """Freeze backbone stages like torchvision does for pretrained backbones."""
    train = list(BACKBONE_LAYERS[:trainable_layers])
    if trainable_layers == len(BACKBONE_LAYERS):
        train.append("bn1")
    for name, parameter in model.backbone.body.named_parameters():
        if not any(name.startswith(layer) for layer in train):
            parameter.requires_grad_(False)


def check_hash_suffix(path: Path) -> None:
    """Verify a torchvision-style ``name-<sha256 prefix>.pth`` file against its content.

    Raises:
        ProjectError: If the name carries a hash prefix that the content does not match.
    """
    match = HASH_SUFFIX.search(path.name)
    if match and not sha256_file(path).startswith(match.group(1)):
        raise ProjectError("pretrained_hash_mismatch", path=path, expected=match.group(1))


def init_from_pretrained(model: Detector, path: Path | None, architecture: str) -> dict:
    """Load local COCO weights into every layer except the class-specific heads.

    Mask heads of a Mask R-CNN checkpoint, the Faster R-CNN box predictor and
    the RetinaNet class logits are skipped.

    Returns:
        Description of the loaded and skipped keys.

    Raises:
        ProjectError: If the weights are absent or do not fit the architecture.
    """
    if path is None or not path.is_file():
        raise ProjectError("pretrained_missing", path=path, architecture=architecture)
    check_hash_suffix(path)
    state = state_dict_of(load_restricted(path), path)
    skip = CLASS_SPECIFIC[architecture]
    kept = {k: v for k, v in state.items() if not k.startswith(skip)}
    result = model.load_state_dict(kept, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(skip)]
    if missing or result.unexpected_keys:
        details = f"missing={missing[:3]}, unexpected={list(result.unexpected_keys)[:3]}"
        raise ProjectError(
            "checkpoint_architecture_mismatch",
            path=path,
            architecture=architecture,
            details=details,
        )
    return {
        "loaded": len(kept),
        "skipped": sorted({".".join(k.split(".")[:3]) for k in state if k.startswith(skip)}),
    }


def save_checkpoint(
    model: nn.Module, path: Path, config: ProjectConfig, extra: dict[str, Any]
) -> None:
    """Save weights with the class order and resize bounds used in training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": {
            "architecture": config.model.architecture,
            "min_size": config.model.min_size,
            "max_size": config.model.max_size,
        },
        "classes": list(config.classes.names),
        **extra,
    }
    torch.save(payload, path)


def load_trained(path: Path, config: ProjectConfig) -> tuple[Detector, dict[str, Any]]:
    """Load an own checkpoint after checking class order and head size.

    Raises:
        ProjectError: If classes are absent, differ from the configuration or the
            head size does not match.
    """
    checkpoint = load_restricted(path)
    classes = checkpoint.get("classes")
    if not classes:
        raise ProjectError("checkpoint_classes_missing", path=path)
    if list(classes) != list(config.classes.names):
        raise ProjectError(
            "checkpoint_classes_mismatch",
            path=path,
            found=list(classes),
            expected=list(config.classes.names),
        )
    stored = (checkpoint.get("config") or {}).get("architecture")
    if stored != config.model.architecture:
        raise ProjectError(
            "checkpoint_architecture_mismatch",
            path=path,
            architecture=config.model.architecture,
            details=f"checkpoint architecture {stored}",
        )
    state = state_dict_of(checkpoint, path)
    outputs = head_outputs(state)
    if outputs != len(classes) + 1:
        raise ProjectError(
            "checkpoint_head_mismatch", path=path, found=outputs, expected=len(classes) + 1
        )
    model = build_detector(config)
    model.load_state_dict(state, strict=True)
    return model.eval(), {k: v for k, v in checkpoint.items() if k != "model"}


def torch_device(name: str) -> torch.device:
    """Return the configured torch device."""
    return torch.device(name)

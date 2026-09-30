"""RetinaNet v2 candidate with ignore regions and a frozen background column.

torchvision RetinaNet predicts one sigmoid score per class column; column
``0`` has no target class. Its weights are set to zero, its bias to
``BACKGROUND_LOGIT`` and its gradients to zero, so its score stays far below
any usable threshold. A plain torchvision RetinaNet (for example in the Pavel
pipeline) loading these weights therefore never returns label ``0``.

Anchors matched to a box with ``IGNORE_LABEL`` are marked "between
thresholds", which excludes them from both the classification and the box
regression loss.
"""

from __future__ import annotations

import torch
from torchvision.models.detection import RetinaNet, retinanet_resnet50_fpn_v2
from torchvision.ops import boxes as box_ops

from sludge_micro.prepare import IGNORE_LABEL

# sigmoid(-20) is about 2e-9, nine orders of magnitude below any grid threshold.
BACKGROUND_LOGIT = -20.0
# Same minimum side as the torchvision Faster R-CNN box head; RetinaNet keeps
# boxes that clipping at the image border reduced to zero height or width.
MIN_BOX_SIDE = 1e-2


class IgnoreAwareRetinaNet(RetinaNet):
    """RetinaNet whose loss skips anchors matched to ignore regions."""

    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Match anchors as torchvision does, then drop matches to ignore regions."""
        matched = []
        between = self.proposal_matcher.BETWEEN_THRESHOLDS
        for anchors_per_image, target in zip(anchors, targets, strict=True):
            if target["boxes"].numel() == 0:
                matched.append(
                    torch.full(
                        (anchors_per_image.size(0),),
                        -1,
                        dtype=torch.int64,
                        device=anchors_per_image.device,
                    )
                )
                continue
            idx = self.proposal_matcher(box_ops.box_iou(target["boxes"], anchors_per_image))
            ignored = (idx >= 0) & (target["labels"][idx.clamp(min=0)] == IGNORE_LABEL)
            matched.append(torch.where(ignored, torch.full_like(idx, between), idx))
        return self.head.compute_loss(targets, head_outputs, anchors, matched)

    def postprocess_detections(
        self,
        head_outputs: dict[str, list[torch.Tensor]],
        anchors: list[list[torch.Tensor]],
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]]:
        """Run torchvision post-processing, drop degenerate boxes, then apply the limit.

        Dropping before the limit keeps the configured number of valid boxes, as
        the pipeline does when it skips zero-area boxes while collecting results.
        """
        limit, self.detections_per_img = self.detections_per_img, torch.iinfo(torch.int64).max
        try:
            detections = super().postprocess_detections(head_outputs, anchors, image_shapes)
        finally:
            self.detections_per_img = limit
        for detection in detections:
            keep = box_ops.remove_small_boxes(detection["boxes"], MIN_BOX_SIDE)[:limit]
            for key in ("boxes", "scores", "labels"):
                detection[key] = detection[key][keep]
        return detections


def background_channels(model: RetinaNet) -> torch.Tensor:
    """Indices of the classification channels that belong to column ``0``."""
    head = model.head.classification_head
    return torch.arange(head.num_anchors) * head.num_classes


def freeze_background(model: RetinaNet) -> None:
    """Pin column ``0`` to a constant very low score and keep it there."""
    conv = model.head.classification_head.cls_logits
    channels = background_channels(model)
    with torch.no_grad():
        conv.weight[channels] = 0.0
        conv.bias[channels] = BACKGROUND_LOGIT

    def zero_rows(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[channels] = 0.0
        return grad

    conv.weight.register_hook(zero_rows)
    conv.bias.register_hook(zero_rows)


def build_retinanet(**kwargs: object) -> RetinaNet:
    """Build RetinaNet v2 without weights, with ignore handling and a frozen background."""
    model = retinanet_resnet50_fpn_v2(**kwargs)
    model.__class__ = IgnoreAwareRetinaNet
    freeze_background(model)
    return model

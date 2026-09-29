from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from cv_module.inference.types import Detection, Segmentation


def read_coco_categories(path: str | Path) -> tuple[str, ...]:
    """Read class names from a COCO instances document.

    Args:
        path: Path to a COCO JSON annotation file.

    Returns:
        Class names ordered by their numeric COCO category identifiers.

    Raises:
        ValueError: If categories are missing, malformed, or duplicated.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    categories = document.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"COCO JSON contains no categories: {path}")
    parsed = []
    for category in categories:
        try:
            category_id, name = int(category["id"]), str(category["name"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid COCO category in {path}: {category!r}") from exc
        if category_id < 1 or not name:
            raise ValueError(f"COCO categories require positive ids and non-empty names: {category!r}")
        parsed.append((category_id, name))
    parsed.sort()
    names = tuple(name for _, name in parsed)
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate category names in {path}")
    return names


def _load_checkpoint(path: Path) -> tuple[dict, dict]:
    """Extract a state dictionary and optional training metadata."""
    import torch
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    metadata = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint does not contain a state dict: {path}")
    state = {str(key).removeprefix("module."): value for key, value in checkpoint.items()}
    return state, metadata if isinstance(metadata, dict) else {}


def _build_model(architecture: str, class_count: int, min_size=None, max_size=None):
    """Construct a supported torchvision detection architecture."""
    try:
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2,
            maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2,
        )
    except ImportError as exc:
        raise RuntimeError("The torchvision backend requires torch and torchvision") from exc
    builders = {
        "maskrcnn_resnet50_fpn_v2": maskrcnn_resnet50_fpn_v2,
        "maskrcnn_resnet50_fpn": maskrcnn_resnet50_fpn,
        "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
        "fasterrcnn_resnet50_fpn": fasterrcnn_resnet50_fpn,
    }
    if architecture not in builders:
        raise ValueError(f"Unsupported torchvision architecture {architecture!r}; choose one of {sorted(builders)}")
    transform = {}
    if min_size is not None:
        transform["min_size"] = int(min_size)
    if max_size is not None:
        transform["max_size"] = int(max_size)
    return builders[architecture](weights=None, weights_backbone=None,
                                  num_classes=class_count + 1, **transform)


class TorchvisionCocoModel:
    """Restore a torchvision model and its class mapping from COCO metadata."""

    def __init__(self, weights: Path, annotations: Path, architecture: str, device: str) -> None:
        """Initialize a COCO-backed torchvision model.

        Args:
            weights: Path to a checkpoint containing a state dictionary.
            annotations: COCO JSON used to recover class ordering.
            architecture: Supported torchvision model-builder name.
            device: Torch device such as ``cpu`` or a CUDA index.

        Raises:
            ValueError: If categories, architecture, or checkpoint are invalid.
            RuntimeError: If torch or torchvision is unavailable.
        """
        import torch
        self.categories = read_coco_categories(annotations)
        self.device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
        state, metadata = _load_checkpoint(weights)
        self.model = _build_model(architecture, len(self.categories),
                                  metadata.get("min_size"), metadata.get("max_size"))
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Checkpoint {weights} is incompatible with {architecture} and "
                             f"{len(self.categories)} COCO categories") from exc
        self.model.to(self.device).eval()

    @property
    def classes(self) -> set[str]:
        """Return class names recovered from the COCO document."""
        return set(self.categories)

    def _predict(self, image_bgr: np.ndarray) -> dict:
        """Convert a BGR image to a tensor and execute the model."""
        import torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        with torch.inference_mode():
            return self.model([tensor])[0]

    def _class_name(self, label: int) -> str:
        """Map a one-based model label to its COCO class name."""
        if not 1 <= label <= len(self.categories):
            raise ValueError(f"Model returned label {label}, outside COCO class mapping")
        return self.categories[label - 1]

    def segment(self, image_bgr: np.ndarray, *, confidence: float,
                max_detections: int, class_name: str) -> list[Segmentation]:
        """Predict instance masks for a selected COCO class.

        Args:
            image_bgr: Source image in OpenCV BGR order.
            confidence: Minimum prediction confidence.
            max_detections: Maximum number of retained masks.
            class_name: COCO class retained as a floc.

        Returns:
            Segmentation predictions in source-image coordinates.

        Raises:
            ValueError: If the configured architecture does not produce masks.
        """
        result = self._predict(image_bgr)
        if "masks" not in result:
            raise ValueError("Configured torchvision floc architecture does not produce masks")
        output = []
        for raw_mask, label, score in zip(
            result["masks"].detach().cpu().numpy()[:, 0],
            result["labels"].detach().cpu().tolist(), result["scores"].detach().cpu().tolist(),
        ):
            name = self._class_name(int(label))
            if score >= confidence and name.casefold() == class_name.casefold():
                mask = raw_mask > 0.5
                if mask.any():
                    output.append(Segmentation(mask, float(score), name))
            if len(output) >= max_detections:
                break
        return output

    def detect(self, image_bgr: np.ndarray, *, confidence: float,
               max_detections: int) -> list[Detection]:
        """Predict object bounding boxes.

        Args:
            image_bgr: Source image in OpenCV BGR order.
            confidence: Minimum prediction confidence.
            max_detections: Maximum number of retained detections.

        Returns:
            Object detections in source-image coordinates.
        """
        result = self._predict(image_bgr)
        output = []
        for box, label, score in zip(
            result["boxes"].detach().cpu().tolist(), result["labels"].detach().cpu().tolist(),
            result["scores"].detach().cpu().tolist(),
        ):
            if score >= confidence:
                output.append(Detection(self._class_name(int(label)), tuple(map(float, box)), float(score)))
            if len(output) >= max_detections:
                break
        return output

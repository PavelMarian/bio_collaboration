from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Segmentation:
    """Represent one predicted instance mask.

    Attributes:
        mask: Boolean mask in source-image coordinates.
        confidence: Model confidence in the range ``[0, 1]``.
        class_name: Human-readable predicted class name.
    """
    mask: np.ndarray
    confidence: float
    class_name: str


@dataclass(frozen=True)
class Detection:
    """Represent one predicted object bounding box.

    Attributes:
        class_name: Human-readable predicted class name.
        bbox_xyxy: Bounding box as ``(x_min, y_min, x_max, y_max)``.
        confidence: Model confidence in the range ``[0, 1]``.
    """
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float


class InferenceBackend(Protocol):
    """Define the backend interface consumed by the analysis pipeline."""

    def segment(self, image_bgr: np.ndarray) -> list[Segmentation]:
        """Return floc segmentations for one BGR image."""
        ...

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Return microorganism detections for one BGR image."""
        ...

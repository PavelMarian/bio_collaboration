from __future__ import annotations

import numpy as np

from cv_module.backends.torchvision import TorchvisionCocoModel
from cv_module.backends.ultralytics import UltralyticsModel
from cv_module.config import ModuleConfig
from cv_module.inference.types import Detection, Segmentation


class ConfigurableBackend:
    """Compose independently configured segmentation and detection models."""

    def __init__(self, config: ModuleConfig) -> None:
        """Create both task-specific model adapters.

        Args:
            config: Valid module configuration describing both models.

        Raises:
            FileNotFoundError: If a required model or COCO JSON is missing.
            ValueError: If model configuration or checkpoint is incompatible.
        """
        config.validate(require_models=True)
        self.config = config
        self._floc = self._create(config.floc_backend, config.floc_model,
                                  config.floc_coco_annotations, config.floc_architecture)
        self._micro = self._create(config.microorganism_backend, config.microorganism_model,
                                   config.microorganism_coco_annotations,
                                   config.microorganism_architecture)

    def _create(self, backend, weights, annotations, architecture):
        """Create one backend-specific model adapter."""
        if backend == "ultralytics":
            return UltralyticsModel(weights)
        return TorchvisionCocoModel(weights, annotations, architecture, self.config.device)

    def segment(self, image_bgr: np.ndarray) -> list[Segmentation]:
        """Run configured floc instance segmentation.

        Args:
            image_bgr: Source image in OpenCV BGR order.

        Returns:
            Floc segmentation predictions.
        """
        common = dict(confidence=self.config.floc_confidence,
                      max_detections=self.config.max_detections,
                      class_name=self.config.floc_class_name)
        if self.config.floc_backend == "ultralytics":
            return self._floc.segment(image_bgr, image_size=self.config.image_size,
                                      iou=self.config.iou_threshold, device=self.config.device,
                                      **common)
        return self._floc.segment(image_bgr, **common)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Run configured microorganism detection.

        Args:
            image_bgr: Source image in OpenCV BGR order.

        Returns:
            Microorganism bounding-box predictions.
        """
        common = dict(confidence=self.config.microorganism_confidence,
                      max_detections=self.config.max_detections)
        if self.config.microorganism_backend == "ultralytics":
            return self._micro.detect(image_bgr, image_size=self.config.image_size,
                                      iou=self.config.iou_threshold, device=self.config.device,
                                      **common)
        return self._micro.detect(image_bgr, **common)

    def model_classes(self) -> tuple[set[str], set[str]]:
        """Return floc and microorganism model class sets.

        Returns:
            Pair containing segmentation classes followed by detection classes.
        """
        return self._floc.classes, self._micro.classes

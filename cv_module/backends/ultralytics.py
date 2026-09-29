from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cv_module.inference.types import Detection, Segmentation


class UltralyticsModel:
    """Adapt one Ultralytics checkpoint to the module prediction types."""

    def __init__(self, weights: Path) -> None:
        """Load an Ultralytics checkpoint.

        Args:
            weights: Path to an Ultralytics ``.pt`` checkpoint.

        Raises:
            RuntimeError: If the ``ultralytics`` package is unavailable.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics model configured but package is not installed") from exc
        self.model = YOLO(str(weights))

    @property
    def classes(self) -> set[str]:
        """Return class names exposed by the loaded checkpoint."""
        return set(map(str, self.model.names.values()))

    def segment(self, image_bgr: np.ndarray, *, image_size: int, confidence: float,
                iou: float, device: str, max_detections: int,
                class_name: str) -> list[Segmentation]:
        """Predict instance masks for a selected class.

        Args:
            image_bgr: Source image in OpenCV BGR order.
            image_size: Ultralytics inference image size.
            confidence: Minimum prediction confidence.
            iou: Non-maximum suppression IoU threshold.
            device: Ultralytics device identifier.
            max_detections: Maximum number of predicted instances.
            class_name: Class retained as a floc segmentation.

        Returns:
            Segmentation predictions in source-image coordinates.
        """
        result = self.model.predict(
            source=image_bgr, imgsz=image_size, conf=confidence, iou=iou,
            device=device, max_det=max_detections, retina_masks=True, verbose=False,
        )[0]
        if result.masks is None:
            return []
        output = []
        for raw_mask, label, score in zip(
            result.masks.data.cpu().numpy(), result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
        ):
            name = str(result.names[int(label)])
            if name.casefold() != class_name.casefold():
                continue
            if raw_mask.shape != image_bgr.shape[:2]:
                raw_mask = cv2.resize(raw_mask, (image_bgr.shape[1], image_bgr.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            mask = raw_mask > 0.5
            if mask.any():
                output.append(Segmentation(mask, float(score), name))
        return output

    def detect(self, image_bgr: np.ndarray, *, image_size: int, confidence: float,
               iou: float, device: str, max_detections: int) -> list[Detection]:
        """Predict object bounding boxes.

        Args:
            image_bgr: Source image in OpenCV BGR order.
            image_size: Ultralytics inference image size.
            confidence: Minimum prediction confidence.
            iou: Non-maximum suppression IoU threshold.
            device: Ultralytics device identifier.
            max_detections: Maximum number of predicted objects.

        Returns:
            Object detections in source-image coordinates.
        """
        result = self.model.predict(
            source=image_bgr, imgsz=image_size, conf=confidence, iou=iou,
            device=device, max_det=max_detections, verbose=False,
        )[0]
        if result.boxes is None:
            return []
        return [
            Detection(str(result.names[int(label)]), tuple(map(float, box)), float(score))
            for box, label, score in zip(
                result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            )
        ]

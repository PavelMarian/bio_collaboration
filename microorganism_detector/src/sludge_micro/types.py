"""Immutable records exchanged between components."""

from __future__ import annotations

from dataclasses import dataclass

BoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class ObjectDetection:
    """One accepted detection in source-image pixel coordinates.

    Attributes:
        class_name: Target class name from the configuration.
        bbox_xyxy: ``(x_min, y_min, x_max, y_max)`` in pixels of the source image.
        confidence: Detector score in ``[0, 1]``; not a calibrated probability.
    """

    class_name: str
    bbox_xyxy: BoxXYXY
    confidence: float


@dataclass(frozen=True)
class ImagePrediction:
    """All accepted detections of one image."""

    image_id: str
    file_name: str
    width: int
    height: int
    objects: tuple[ObjectDetection, ...]


@dataclass(frozen=True)
class ImageFacts:
    """Decoded properties and fingerprints of one image file."""

    width: int
    height: int
    channels: int
    file_sha256: str
    pixel_sha256: str
    dhash: str
    exif_datetime_original: str | None
    camera_model: str | None


@dataclass(frozen=True)
class BoxCheck:
    """Validated box and the correction applied to it, if any."""

    box: BoxXYXY | None
    issue: str | None
    fix: str | None

"""Passport of a delivered model: the contract that inference checks before use.

A passport is a small JSON document kept next to the code. It names the
architecture, the target classes in label order, image handling, detector
post-processing, the confidence threshold chosen on validation, the SHA-256 and
size of the weights file, the training settings and the evaluation results of
the run that produced the weights. ``passport_predictor`` refuses weights whose
checksum, classes, architecture, head or resize bounds differ from the
passport, so a delivered model runs without the folders of its training run.
``build_passport`` writes a passport from a run of this package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from sludge_micro.archives import sha256_file
from sludge_micro.checkpoint import head_outputs, load_restricted, state_dict_of
from sludge_micro.infer import Predictor
from sludge_micro.messages import ProjectError
from sludge_micro.model import POSTPROCESS_KEYS, Detector, build_detector_raw, torch_device
from sludge_micro.reporting import read_json

FORMAT = "sludge_micro/passport/1"
# File names of delivered weights, the ones the Pavel pipeline block refers to.
WEIGHTS_NAMES = {
    "torchvision": "microorganisms_fasterrcnn.pt",
    "ultralytics": "microorganisms_yolo.pt",
}


class Record(BaseModel):
    """Immutable passport node that rejects unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class PassportClass(Record):
    """One target class and its model label."""

    label: int = Field(ge=1)
    name: str
    title: str


class ImageHandling(Record):
    """How an image reaches the detector."""

    formats: tuple[str, ...]
    color: Literal["opencv_bgr_uint8_to_rgb_float"]
    value_range: tuple[float, float]
    expected_width: int = Field(gt=0)
    expected_height: int = Field(gt=0)
    min_size: int = Field(gt=0)
    max_size: int = Field(gt=0)


class DetectorSettings(Record):
    """Architecture, post-processing and the confidence threshold of accepted detections."""

    architecture: Literal["fasterrcnn_resnet50_fpn_v2", "retinanet_resnet50_fpn_v2"]
    min_score: float = Field(ge=0, le=1)
    nms_iou: float = Field(gt=0, le=1)
    max_detections: int = Field(gt=0)
    confidence: float = Field(gt=0, le=1)


class WeightsFile(Record):
    """Delivered weights file and its checksum."""

    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)
    checkpoint_format: str


class ModelPassport(Record):
    """Contract of a delivered model; the loose sections describe how it was obtained."""

    format: Literal["sludge_micro/passport/1"]
    run_id: str
    status: str
    classes: tuple[PassportClass, ...]
    image: ImageHandling
    detector: DetectorSettings
    weights: WeightsFile
    training: dict[str, Any]
    evaluation: dict[str, Any]
    integration: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> ModelPassport:
        if [c.label for c in self.classes] != list(range(1, len(self.classes) + 1)):
            raise ValueError("class labels must be 1..N in listed order")
        return self

    @property
    def names(self) -> tuple[str, ...]:
        """Class names in label order."""
        return tuple(c.name for c in self.classes)


def load_passport(path: Path) -> ModelPassport:
    """Read and check a passport.

    Raises:
        ProjectError: If the file is missing or does not match the passport schema.
    """
    if not path.is_file():
        raise ProjectError("file_missing", path=path, action="укажите файл паспорта модели.")
    try:
        return ModelPassport.model_validate(read_json(path))
    except ValidationError as exc:
        raise ProjectError("passport_invalid", path=path, details=exc) from exc


def check_weights(passport: ModelPassport, weights: Path) -> None:
    """Refuse a weights file whose size or SHA-256 differs from the passport.

    Raises:
        ProjectError: If the file is missing or differs from the passport.
    """
    if not weights.is_file():
        raise ProjectError("file_missing", path=weights, action="загрузите веса модели.")
    found = sha256_file(weights)
    if found != passport.weights.sha256 or weights.stat().st_size != passport.weights.bytes:
        raise ProjectError(
            "weights_mismatch", path=weights, expected=passport.weights.sha256, found=found
        )


def check_checkpoint(passport: ModelPassport, checkpoint: dict[str, Any], path: Path) -> None:
    """Refuse a checkpoint whose classes, architecture, resize or head differ from the passport.

    Raises:
        ProjectError: If any of these differs.
    """
    config = checkpoint.get("config") or {}
    image = passport.image
    found = {
        "classes": list(checkpoint.get("classes") or []),
        "architecture": config.get("architecture"),
        "min_size": config.get("min_size"),
        "max_size": config.get("max_size"),
        "head_outputs": head_outputs(state_dict_of(checkpoint, path)),
    }
    expected = {
        "classes": list(passport.names),
        "architecture": passport.detector.architecture,
        "min_size": image.min_size,
        "max_size": image.max_size,
        "head_outputs": len(passport.classes) + 1,
    }
    differs = sorted(k for k in expected if found[k] != expected[k])
    if differs:
        raise ProjectError("passport_checkpoint_mismatch", path=path, fields=", ".join(differs))


def passport_model(passport: ModelPassport, weights: Path) -> Detector:
    """Build the detector the passport describes and load the checked weights into it."""
    check_weights(passport, weights)
    checkpoint = load_restricted(weights)
    check_checkpoint(passport, checkpoint, weights)
    d = passport.detector
    values = (d.min_score, d.nms_iou, d.max_detections)
    model = build_detector_raw(
        d.architecture,
        len(passport.classes) + 1,
        passport.image.min_size,
        passport.image.max_size,
        **dict(zip(POSTPROCESS_KEYS[d.architecture], values, strict=True)),
    )
    model.load_state_dict(state_dict_of(checkpoint, weights), strict=True)
    return model.eval()


def passport_predictor(passport: ModelPassport, weights: Path, device: str = "cpu") -> Predictor:
    """Predictor of the delivered model at the passport's confidence threshold."""
    model = passport_model(passport, weights)
    return Predictor(model, passport.names, passport.detector.confidence, torch_device(device))

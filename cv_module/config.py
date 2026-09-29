from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CLASSES = (
    "attached_ciliates",
    "filamentous_bacteria",
    "rotifers",
    "testate_amoebae",
    "free_swimming_ciliates",
    "nematoda",
    "gastrotrichs",
)


@dataclass(frozen=True)
class ModuleConfig:
    """Describe model adapters and inference parameters.

    Attributes:
        floc_model: Path to the floc segmentation checkpoint.
        microorganism_model: Path to the microorganism detection checkpoint.
        floc_backend: Backend used for floc segmentation.
        microorganism_backend: Backend used for microorganism detection.
        floc_architecture: Torchvision architecture for the floc model.
        microorganism_architecture: Torchvision architecture for the detector.
        floc_coco_annotations: COCO JSON used to map floc class labels.
        microorganism_coco_annotations: COCO JSON used to map detector labels.
        device: Inference device such as ``cpu`` or a CUDA index.
        image_size: Ultralytics inference image size in pixels.
        floc_confidence: Minimum segmentation confidence.
        microorganism_confidence: Minimum detection confidence.
        iou_threshold: Non-maximum suppression IoU threshold.
        max_detections: Maximum number of results per image and task.
        floc_class_name: Expected floc class name.
        microorganism_classes: Classes included in aggregate output tables.
    """
    floc_model: Path
    microorganism_model: Path
    floc_backend: str = "ultralytics"
    microorganism_backend: str = "ultralytics"
    floc_architecture: str = "maskrcnn_resnet50_fpn_v2"
    microorganism_architecture: str = "fasterrcnn_resnet50_fpn_v2"
    floc_coco_annotations: Path | None = None
    microorganism_coco_annotations: Path | None = None
    device: str = "cpu"
    image_size: int = 960
    floc_confidence: float = 0.25
    microorganism_confidence: float = 0.25
    iou_threshold: float = 0.60
    max_detections: int = 300
    floc_class_name: str = "floc"
    microorganism_classes: tuple[str, ...] = DEFAULT_CLASSES

    def validate(self, require_models: bool = True) -> None:
        """Validate configuration values and optionally referenced files.

        Args:
            require_models: Whether model and COCO annotation files must exist.

        Raises:
            FileNotFoundError: If a required model or annotation file is absent.
            ValueError: If a backend or numeric parameter is invalid.
        """
        if require_models:
            for label, path in (
                ("floc segmentation", self.floc_model),
                ("microorganism detection", self.microorganism_model),
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing {label} model: {path}")
            for label, backend, annotations in (
                ("floc", self.floc_backend, self.floc_coco_annotations),
                ("microorganism", self.microorganism_backend, self.microorganism_coco_annotations),
            ):
                if backend == "torchvision" and (annotations is None or not annotations.is_file()):
                    raise FileNotFoundError(f"{label} torchvision backend requires a COCO annotations JSON: {annotations}")
        allowed = {"ultralytics", "torchvision"}
        if self.floc_backend not in allowed or self.microorganism_backend not in allowed:
            raise ValueError(f"Model backend must be one of {sorted(allowed)}")
        if self.image_size <= 0 or self.max_detections <= 0:
            raise ValueError("image_size and max_detections must be positive")
        for name, value in (
            ("floc_confidence", self.floc_confidence),
            ("microorganism_confidence", self.microorganism_confidence),
            ("iou_threshold", self.iou_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if len(set(self.microorganism_classes)) != len(self.microorganism_classes):
            raise ValueError("microorganism_classes contains duplicates")


def _model_path(value: Any, base_dir: Path, field: str) -> Path:
    """Resolve a required model path relative to the configuration file."""
    if not value:
        raise ValueError(f"models.{field} is required")
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _optional_path(value: Any, base_dir: Path) -> Path | None:
    """Resolve an optional path relative to the configuration file."""
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _model_block(models: dict[str, Any], name: str) -> dict[str, Any]:
    """Normalize a scalar model path or an extended model mapping.

    Args:
        models: Raw ``models`` section from YAML.
        name: Model field to normalize.

    Returns:
        A model configuration mapping containing at least a backend and weights.
    """
    value = models.get(name)
    return value if isinstance(value, dict) else {"weights": value, "backend": "ultralytics"}


def load_config(path: str | Path) -> ModuleConfig:
    """Load module configuration from YAML.

    Relative model and annotation paths are resolved from the YAML directory.

    Args:
        path: Path to the configuration file.

    Returns:
        Parsed and value-validated module configuration.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If required fields or parameter values are invalid.
        yaml.YAMLError: If the file is not valid YAML.
    """
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    models = raw.get("models") or {}
    inference = raw.get("inference") or {}
    classes = raw.get("microorganism_classes", DEFAULT_CLASSES)
    if not isinstance(classes, list | tuple) or not all(isinstance(v, str) for v in classes):
        raise ValueError("microorganism_classes must be a list of strings")
    floc = _model_block(models, "floc_segmentation")
    micro = _model_block(models, "microorganism_detection")
    config = ModuleConfig(
        floc_model=_model_path(floc.get("weights"), config_path.parent, "floc_segmentation.weights"),
        microorganism_model=_model_path(micro.get("weights"), config_path.parent, "microorganism_detection.weights"),
        floc_backend=str(floc.get("backend", "ultralytics")).lower(),
        microorganism_backend=str(micro.get("backend", "ultralytics")).lower(),
        floc_architecture=str(floc.get("architecture", "maskrcnn_resnet50_fpn_v2")),
        microorganism_architecture=str(micro.get("architecture", "fasterrcnn_resnet50_fpn_v2")),
        floc_coco_annotations=_optional_path(floc.get("coco_annotations"), config_path.parent),
        microorganism_coco_annotations=_optional_path(micro.get("coco_annotations"), config_path.parent),
        device=str(inference.get("device", "cpu")),
        image_size=int(inference.get("image_size", 960)),
        floc_confidence=float(inference.get("floc_confidence", 0.25)),
        microorganism_confidence=float(inference.get("microorganism_confidence", 0.25)),
        iou_threshold=float(inference.get("iou_threshold", 0.60)),
        max_detections=int(inference.get("max_detections", 300)),
        floc_class_name=str(inference.get("floc_class_name", "floc")),
        microorganism_classes=tuple(classes),
    )
    config.validate(require_models=False)
    return config

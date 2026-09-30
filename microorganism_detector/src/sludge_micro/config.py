"""Validated project configuration.

All thresholds, weights and paths live in YAML. Paths are relative to the
project root, which is the nearest ancestor of the configuration file that
contains ``pyproject.toml``.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from sludge_micro.messages import ProjectError


class Frozen(BaseModel):
    """Immutable configuration node that rejects unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class PathsConfig(Frozen):
    """Project-relative locations of inputs and outputs."""

    raw_dir: Path
    processed_dir: Path
    experiments_dir: Path
    outputs_dir: Path
    # Optional inputs of the data audit: the requirements document, its machine registry and
    # a received reference experiment. Without them the audit leaves those sections out.
    specification: Path | None = None
    requirements_registry: Path | None = None
    reference_checkpoint: Path | None = None
    reference_metrics: Path | None = None
    # COCO weights that training starts from; inference of a trained model never needs them.
    pretrained_weights: Path | None = None
    # Folder of a locked split that another stage owns and this configuration only reads;
    # ``None`` means the split lives in ``experiments_dir``.
    split_dir: Path | None = None
    # Folder of the Pavel pipeline (the one that holds ``cv_module``) for hand-over checks.
    pavel_pipeline: Path | None = None


class SourceConfig(Frozen):
    """One raw archive and its declared role."""

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    archive: str
    format: Literal["coco", "yolo", "voc"]
    role: Literal["microorganisms", "floc", "external"]
    priority: int = Field(ge=0)


class TargetClass(Frozen):
    """Explicit link between output name, source category and model label."""

    name: str
    title: str
    source_category: str
    source_category_id: int = Field(ge=1)
    label: int = Field(ge=1)


class ClassesConfig(Frozen):
    """Target classes and the treatment of every other source category."""

    target: tuple[TargetClass, ...]
    background: tuple[str, ...]
    ignore: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    not_detected: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> ClassesConfig:
        labels = [c.label for c in self.target]
        if labels != list(range(1, len(labels) + 1)):
            raise ValueError("target labels must be 1..N in listed order")
        names = [c.name for c in self.target]
        groups = [names, self.background, self.ignore, self.out_of_scope, self.not_detected]
        flat = [name for group in groups for name in group]
        if len(set(flat)) != len(flat):
            raise ValueError("class names must be unique across all groups")
        ids = [c.source_category_id for c in self.target]
        if len(set(ids)) != len(ids):
            raise ValueError("target source_category_id values must be unique")
        return self

    @property
    def names(self) -> tuple[str, ...]:
        """Target class names in model label order."""
        return tuple(c.name for c in self.target)

    def role_of(self, category: str) -> str | None:
        """Return the declared role of a source category name."""
        for role, names in (
            ("target", tuple(c.source_category for c in self.target)),
            ("background", self.background),
            ("ignore", self.ignore),
            ("out_of_scope", self.out_of_scope),
            ("not_detected", self.not_detected),
        ):
            if category in names:
                return role
        return None


class ExternalConfig(Frozen):
    """Optional external dataset; disabled unless explicitly enabled."""

    enabled: bool
    source_id: str
    class_map: dict[str, str]


class AuditConfig(Frozen):
    """Checks applied to raw archives."""

    image_suffixes: tuple[str, ...]
    expected_width: int = Field(gt=0)
    expected_height: int = Field(gt=0)
    dhash_size: int = Field(ge=8)
    near_duplicate_max_hamming: int = Field(ge=0)
    duplicate_match_iou: float = Field(gt=0, le=1)
    bbox_tolerance_px: float = Field(ge=0)


class PrepareConfig(Frozen):
    """Policies that turn audited annotations into supervision."""

    ignore_policy: Literal["ignore_label", "exclude_image"]
    unconfirmed_empty_policy: Literal["exclude", "negative"]
    conflicting_duplicate_policy: Literal["exclude_group"]
    bbox_policy: Literal["coco_bbox_then_polygon"]
    # Research runs may read a duplicate-handling variant instead of the working set.
    variant: str | None = None


class GroupingConfig(Frozen):
    """Rules that join related frames for the non-temporal group split."""

    orb_min_inliers: int = Field(gt=0)
    orb_features: int = Field(gt=0)
    orb_ratio: float = Field(gt=0, lt=1)
    orb_ransac_px: float = Field(gt=0)
    orb_image_width: int = Field(gt=0)
    filename_series: bool
    sampling_point_tokens: bool
    unmarked_per_session: bool
    search_iterations: int = Field(gt=0)


class SplitConfig(Frozen):
    """Split protocol written before any measurement.

    ``temporal`` is the working protocol. ``group`` is a non-temporal protocol
    allowed only with a recorded owner approval.
    """

    protocol: Literal["temporal", "group"]
    confirmed_time_sources: tuple[Literal["exif_datetime_original", "owner_metadata"], ...]
    owner_metadata: Path | None = None
    train_end: datetime | None = None
    validation_end: datetime | None = None
    boundary_group_policy: Literal["group_to_later_split"]
    target_fractions: dict[Literal["train", "validation", "test"], float]
    alternative_approval: str | None = None
    grouping: GroupingConfig | None = None

    @model_validator(mode="after")
    def _check(self) -> SplitConfig:
        values = self.target_fractions
        if set(values) != {"train", "validation", "test"} or abs(sum(values.values()) - 1) > 1e-9:
            raise ValueError("target_fractions must cover train, validation and test and sum to 1")
        if min(values.values()) <= 0:
            raise ValueError("target_fractions must be positive")
        if self.protocol == "group" and (not self.alternative_approval or self.grouping is None):
            raise ValueError("group protocol needs alternative_approval and grouping")
        return self


class YoloConfig(Frozen):
    """Settings of the Ultralytics YOLO candidate."""

    weights: Path | None = None
    image_size: int = Field(gt=0)
    ignore_policy: Literal["exclude_image"]
    batch: int = Field(default=8, gt=0)
    nominal_batch: int = Field(default=8, gt=0)
    optimizer: Literal["SGD"] = "SGD"
    lr0: float = Field(default=0.01, gt=0)
    final_lr_factor: float = Field(default=0.01, gt=0, le=1)
    momentum: float = Field(default=0.937, ge=0, lt=1)
    weight_decay: float = Field(default=0.0005, ge=0)
    warmup_epochs: float = Field(default=3.0, ge=0)


TORCHVISION_ARCHITECTURES = ("fasterrcnn_resnet50_fpn_v2", "retinanet_resnet50_fpn_v2")


class ModelConfig(Frozen):
    """Detector architecture and resize bounds."""

    architecture: Literal[
        "fasterrcnn_resnet50_fpn_v2", "retinanet_resnet50_fpn_v2", "ultralytics_yolo"
    ]
    min_size: int = Field(gt=0)
    max_size: int = Field(gt=0)
    trainable_backbone_layers: int = Field(ge=0, le=5)
    yolo: YoloConfig | None = None

    @model_validator(mode="after")
    def _check(self) -> ModelConfig:
        if (self.architecture == "ultralytics_yolo") != (self.yolo is not None):
            raise ValueError("model.yolo is required for ultralytics_yolo and only for it")
        return self

    @property
    def backend(self) -> str:
        """Return ``torchvision`` or ``ultralytics``."""
        return "torchvision" if self.architecture in TORCHVISION_ARCHITECTURES else "ultralytics"


class AugmentConfig(Frozen):
    """Train-only augmentations, each enabled explicitly."""

    rotation90: bool
    hflip: bool
    vflip: bool
    flip_probability: float = Field(ge=0, le=1)


class TrainConfig(Frozen):
    """Optimisation settings of one run."""

    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    accumulate: int = Field(gt=0)
    lr: float = Field(gt=0)
    momentum: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    sampling: Literal["none", "rarest_class_presence"]
    augment: AugmentConfig
    # Leave out training images that contain ignore regions (validation and test keep them).
    exclude_ignore_images: bool = False
    # BatchNorm while torchvision detectors train: ``batch`` normalises with the statistics of
    # each training batch and updates the stored ones (the regime of the first research runs);
    # ``frozen`` keeps every BatchNorm layer on its stored statistics, as inference does.
    batchnorm: Literal["batch", "frozen"] = "batch"


class GridConfig(Frozen):
    """Inclusive grid of candidate confidence thresholds."""

    start: float = Field(ge=0, le=1)
    stop: float = Field(ge=0, le=1)
    step: float = Field(gt=0)


class EvaluationConfig(Frozen):
    """Matching and metric rules fixed before measurement."""

    iou_threshold: float = Field(gt=0, le=1)
    ignore_ioa_threshold: float = Field(gt=0, le=1)
    matching: Literal["greedy_by_confidence"]
    unsupported_class_policy: Literal["exclude_if_no_support_and_no_predictions"]
    confidence_grid: GridConfig
    selection_tie_break: Literal["higher_confidence"]


class PostprocessConfig(Frozen):
    """Detector post-processing shared by evaluation and inference."""

    min_score: float = Field(ge=0, le=1)
    nms_iou: float = Field(gt=0, le=1)
    max_detections: int = Field(gt=0)


class ProtocolConfig(Frozen):
    """Acceptance rule and compute budget for hypothesis checks."""

    primary_metric: Literal["macro_f1"]
    acceptance: Literal["candidate_greater_than_control"]
    per_class_constraints: None = None
    max_runs: int = Field(gt=0)
    max_epochs_per_run: int = Field(gt=0)
    # Run id of the control; ``None`` means the only run with the configured architecture.
    control_run: str | None = None
    # Experiments folder of the control when it belongs to an earlier stage.
    control_dir: Path | None = None

    @model_validator(mode="after")
    def _check(self) -> ProtocolConfig:
        if self.control_dir is not None and self.control_run is None:
            raise ValueError("protocol.control_dir needs protocol.control_run")
        return self


class RuntimeConfig(Frozen):
    """Seed, threads and device."""

    seed: int = Field(ge=0)
    threads: int = Field(gt=0)
    loader_workers: int = Field(ge=0)
    device: Literal["cpu", "mps", "cuda"]
    deterministic: bool


class ProjectConfig(Frozen):
    """Root configuration object."""

    version: Literal[1]
    paths: PathsConfig
    sources: tuple[SourceConfig, ...]
    classes: ClassesConfig
    external: ExternalConfig
    audit: AuditConfig
    prepare: PrepareConfig
    split: SplitConfig
    model: ModelConfig
    train: TrainConfig
    evaluation: EvaluationConfig
    postprocess: PostprocessConfig
    protocol: ProtocolConfig
    runtime: RuntimeConfig
    duplicate_variants: dict[str, str | None] = Field(
        default_factory=lambda: {"exclude_conflicts": None}
    )
    root: Path = Path(".")

    @model_validator(mode="after")
    def _check(self) -> ProjectConfig:
        ids = [s.id for s in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError("source ids must be unique")
        if self.train.epochs > self.protocol.max_epochs_per_run:
            raise ValueError("train.epochs exceeds protocol.max_epochs_per_run")
        micro = {s.id for s in self.sources if s.role == "microorganisms"}
        if self.duplicate_variants.get("exclude_conflicts", "missing") is not None:
            raise ValueError("duplicate_variants must contain exclude_conflicts: null")
        if any(v is not None and v not in micro for v in self.duplicate_variants.values()):
            raise ValueError("duplicate_variants must prefer a microorganism source")
        return self

    def resolve(self, path: Path | str) -> Path:
        """Return an absolute path for a project-relative path without following symlinks."""
        return Path(os.path.abspath(self.root / Path(path)))

    def source(self, source_id: str) -> SourceConfig:
        """Return the source with the given identifier."""
        return next(s for s in self.sources if s.id == source_id)


def find_project_root(path: Path) -> Path:
    """Find the nearest ancestor directory that contains ``pyproject.toml``.

    Raises:
        ProjectError: If no such directory exists.
    """
    absolute = Path(os.path.abspath(path))
    for candidate in (absolute.parent, *absolute.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ProjectError("project_root_missing", path=path)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested mappings; values of ``override`` win."""
    out = dict(base)
    for key, value in override.items():
        both = isinstance(value, dict) and isinstance(out.get(key), dict)
        out[key] = deep_merge(out[key], value) if both else value
    return out


def read_raw(path: Path) -> dict[str, Any]:
    """Read YAML and resolve an optional ``extends`` chain relative to the file.

    Raises:
        ProjectError: If a file in the chain is missing.
    """
    if not path.is_file():
        raise ProjectError("file_missing", path=path, action="укажите путь к YAML.")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = raw.pop("extends", None)
    return deep_merge(read_raw((path.parent / base).resolve()), raw) if base else raw


def load_config(path: Path | str) -> ProjectConfig:
    """Load and validate the project configuration.

    Args:
        path: YAML configuration file.

    Returns:
        Validated immutable configuration with the resolved project root.

    Raises:
        ProjectError: If the file is missing or fails schema validation.
    """
    config_path = Path(path)
    raw = read_raw(config_path)
    raw["root"] = find_project_root(config_path)
    try:
        return ProjectConfig.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ProjectError("config_invalid", path=config_path, details=details) from exc

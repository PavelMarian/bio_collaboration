"""Ultralytics YOLO candidate: dataset export, predictor and training entry point.

The ``ultralytics`` package and YOLO weights are not part of the locked
environment. The package is imported lazily; without it every entry point
fails with a message that names the missing resource. YOLO has no ignore
regions, so images with ``unknown`` or ``custom_microorganism`` areas are
left out of the YOLO dataset (``model.yolo.ignore_policy: exclude_image``).
Ultralytics augmentations are switched off so that the candidate differs
from the control run only in the detector.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sludge_micro.config import ProjectConfig
from sludge_micro.imaging import check_bgr_contract
from sludge_micro.infer import accept
from sludge_micro.messages import ProjectError
from sludge_micro.types import ObjectDetection

SPLIT_DIRS = {"train": "train", "validation": "val", "test": "test"}
NO_AUGMENT = {
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "erasing": 0.0,
}


def load_yolo(weights: Path) -> Any:  # noqa: ANN401
    """Load an Ultralytics model from a local file.

    Raises:
        ProjectError: If the package or the weights file is missing.
    """
    os.environ.setdefault("YOLO_OFFLINE", "true")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ProjectError("ultralytics_missing") from exc
    if not weights.is_file():
        raise ProjectError("pretrained_missing", path=weights, architecture="ultralytics_yolo")
    return YOLO(str(weights))


def ultralytics_version() -> str | None:
    """Installed Ultralytics version, if any."""
    from importlib import metadata

    try:
        return metadata.version("ultralytics")
    except metadata.PackageNotFoundError:
        return None


def apply_threads(threads: int) -> None:
    """Make Ultralytics CPU training use the configured thread count.

    ``select_device`` resets torch threads to the module constant ``NUM_THREADS``
    (the core count minus one, at most eight); the constant is set to the
    configured value so that every run of the project uses the same count.
    """
    import importlib

    for name in ("ultralytics.utils", "ultralytics.utils.torch_utils"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        module.NUM_THREADS = threads


def model_names(model: Any) -> tuple[str, ...]:  # noqa: ANN401
    """Return class names of a YOLO model in index order."""
    names = model.names
    return tuple(names[i] for i in sorted(names)) if isinstance(names, dict) else tuple(names)


class UltralyticsPredictor:
    """YOLO model behind the same interface as the torchvision predictor."""

    def __init__(self, weights: Path, config: ProjectConfig, confidence: float) -> None:
        """Load the model and check that its classes follow the configured order."""
        self.model = load_yolo(weights)
        apply_threads(config.runtime.threads)
        self.classes = config.classes.names
        found = model_names(self.model)
        if found != self.classes:
            raise ProjectError(
                "checkpoint_classes_mismatch",
                path=weights,
                found=list(found),
                expected=list(self.classes),
            )
        self.confidence = float(confidence)
        self.settings = {
            "imgsz": config.model.yolo.image_size,
            "conf": config.postprocess.min_score,
            "iou": config.postprocess.nms_iou,
            "max_det": config.postprocess.max_detections,
            "device": config.runtime.device,
            "verbose": False,
        }

    def raw(self, image_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """Return boxes, one-based labels and scores above the minimum score."""
        check_bgr_contract(image_bgr)
        result = self.model.predict(source=image_bgr.copy(), **self.settings)[0]
        if result.boxes is None or len(result.boxes.cls) == 0:
            return {"boxes": np.zeros((0, 4)), "labels": np.zeros(0, int), "scores": np.zeros(0)}
        return {
            "boxes": result.boxes.xyxy.cpu().numpy().astype(float),
            "labels": result.boxes.cls.cpu().numpy().astype(int) + 1,
            "scores": result.boxes.conf.cpu().numpy().astype(float),
        }

    def predict_bgr(self, image_bgr: np.ndarray) -> tuple[ObjectDetection, ...]:
        """Return accepted detections of one image at the configured confidence."""
        return accept(self.raw(image_bgr), image_bgr.shape, self.classes, self.confidence)


def label_lines(rows: pd.DataFrame, width: int, height: int) -> list[str]:
    """YOLO box lines ``class cx cy w h`` normalised by the image size."""
    lines = []
    for r in rows.itertuples():
        cx, cy = (r.x1 + r.x2) / 2 / width, (r.y1 + r.y2) / 2 / height
        w, h = (r.x2 - r.x1) / width, (r.y2 - r.y1) / height
        lines.append(f"{int(r.label) - 1} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def export_yolo_dataset(
    config: ProjectConfig,
    images: pd.DataFrame,
    anns: pd.DataFrame,
    splits: dict[str, str],
    out_dir: Path,
) -> dict[str, Any]:
    """Write images, labels and ``data.yaml`` in the Ultralytics layout."""
    ignored = set(anns.loc[anns["supervision"] == "ignore", "image_id"])
    counts = {name: 0 for name in SPLIT_DIRS}
    for row in images.sort_values("image_id", kind="stable").itertuples():
        split = splits.get(row.image_id)
        if split is None or row.image_id in ignored:
            continue
        source = config.resolve(row.path)
        target = out_dir / "images" / SPLIT_DIRS[split] / f"{row.image_id}{source.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows = anns[(anns["image_id"] == row.image_id) & (anns["supervision"] == "target")]
        label = out_dir / "labels" / SPLIT_DIRS[split] / f"{row.image_id}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(label_lines(rows, int(row.width), int(row.height))) + "\n")
        counts[split] += 1
    data = {
        "path": str(out_dir),
        **{v: f"images/{v}" for v in SPLIT_DIRS.values()},
        "names": dict(enumerate(config.classes.names)),
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return {"images": counts, "excluded_with_ignore": len(ignored & set(splits))}


def train_arguments(config: ProjectConfig, data_yaml: Path, out_dir: Path) -> dict[str, Any]:
    """Ultralytics training arguments that mirror the configured protocol."""
    augment, yolo = config.train.augment, config.model.yolo
    if augment.rotation90:
        raise ProjectError("yolo_rotation_unsupported")
    flips = {
        "fliplr": augment.flip_probability if augment.hflip else 0.0,
        "flipud": augment.flip_probability if augment.vflip else 0.0,
    }
    return (
        NO_AUGMENT
        | flips
        | {
            "data": str(data_yaml),
            "epochs": config.train.epochs,
            "batch": yolo.batch,
            "nbs": yolo.nominal_batch,
            "imgsz": yolo.image_size,
            "seed": config.runtime.seed,
            "deterministic": True,
            "workers": config.runtime.loader_workers,
            "device": config.runtime.device,
            "optimizer": yolo.optimizer,
            "lr0": yolo.lr0,
            "lrf": yolo.final_lr_factor,
            "momentum": yolo.momentum,
            "weight_decay": yolo.weight_decay,
            "warmup_epochs": yolo.warmup_epochs,
            "amp": False,
            "project": str(out_dir),
            "name": "ultralytics",
            "exist_ok": True,
            "plots": False,
            "val": False,
        }
    )


def yolo_model_for(config: ProjectConfig, run_id: str, resume: bool) -> Any:  # noqa: ANN401
    """Pretrained model for a new run, or the run's ``last.pt`` for ``--resume``.

    Raises:
        ProjectError: If a new run would overwrite results or the resume state is absent.
    """
    from sludge_micro.reporting import relative
    from sludge_micro.runs import run_dirs

    report_dir, out_dir = run_dirs(config, run_id)
    last = out_dir / "ultralytics" / "weights" / "last.pt"
    if resume:
        if not last.is_file():
            raise ProjectError("resume_missing", path=relative(last, config.root))
        return load_yolo(last)
    if (out_dir / "model.pt").exists() or (report_dir / "metrics.json").exists() or last.exists():
        raise ProjectError("run_exists", run_id=run_id, path=relative(out_dir, config.root))
    if config.model.yolo.weights is None:
        raise ProjectError("pretrained_missing", path=None, architecture="ultralytics_yolo")
    return load_yolo(config.resolve(config.model.yolo.weights))


def run_train_yolo(config: ProjectConfig, run_id: str, resume: bool = False) -> dict[str, Any]:
    """Train the YOLO candidate on the locked split and evaluate it on validation.

    ``resume`` continues from the run's ``last.pt`` with the Ultralytics resume
    mechanism, which restores the epoch, optimizer and EMA state.
    """
    from sludge_micro.prepare import load_prepared
    from sludge_micro.runs import evaluate_run, run_dirs, write_run_manifest
    from sludge_micro.runtime import apply_runtime, code_at_start
    from sludge_micro.split import load_split
    from sludge_micro.train import check_budget

    started = code_at_start(config.root)
    manifest = load_split(config)
    check_budget(config, run_id)
    if config.train.batchnorm != "batch":
        raise ProjectError("yolo_batchnorm_unsupported", mode=config.train.batchnorm)
    model = yolo_model_for(config, run_id, resume)
    apply_threads(config.runtime.threads)
    apply_runtime(config.runtime)
    report_dir, out_dir = run_dirs(config, run_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    images, anns = load_prepared(config)
    splits = dict(zip(manifest["image_id"], manifest["split"], strict=True))
    dataset = export_yolo_dataset(config, images, anns, splits, out_dir / "yolo_dataset")
    if resume:
        model.train(resume=True)
    else:
        model.train(**train_arguments(config, out_dir / "yolo_dataset" / "data.yaml", out_dir))
    shutil.copyfile(out_dir / "ultralytics" / "weights" / "last.pt", out_dir / "model.pt")
    metrics = evaluate_run(config, run_id, "validation")
    write_run_manifest(
        config,
        run_id,
        {
            "code_at_start": started,
            "yolo_dataset": dataset,
            "resumed": resume,
            "ultralytics_version": ultralytics_version(),
        },
    )
    return metrics


def preflight_yolo(config: ProjectConfig) -> dict[str, Any]:
    """Load YOLO weights and export the training data with the configured class order."""
    import tempfile

    from sludge_micro.prepare import load_prepared
    from sludge_micro.split import load_split

    if config.model.yolo.weights is None:
        raise ProjectError("pretrained_missing", path=None, architecture="ultralytics_yolo")
    model = load_yolo(config.resolve(config.model.yolo.weights))
    manifest = load_split(config)
    images, anns = load_prepared(config)
    splits = dict(zip(manifest["image_id"], manifest["split"], strict=True))
    with tempfile.TemporaryDirectory() as folder:
        report = export_yolo_dataset(config, images, anns, splits, Path(folder))
        names = yaml.safe_load((Path(folder) / "data.yaml").read_text(encoding="utf-8"))["names"]
    return {
        "pretrained_classes": len(model_names(model)),
        "dataset": report,
        "dataset_names_match": list(names.values()) == list(config.classes.names),
    }

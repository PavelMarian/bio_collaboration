from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from cv_module.config import ModuleConfig, load_config
from cv_module.inference.backend import ConfigurableBackend
from cv_module.inference.types import InferenceBackend
from cv_module.masks.features import (
    FEATURE_COLUMNS,
    FLOC_CSV_COLUMNS,
    aggregate_rows,
    extract_features,
    instance_row,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".bmp"}


def read_image(path: Path) -> np.ndarray:
    """Read an image from a Unicode-safe filesystem path.

    Args:
        path: JPG or BMP image path.

    Returns:
        Decoded image in OpenCV BGR order.

    Raises:
        ValueError: If OpenCV cannot decode the file.
    """
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def discover_images(source: str | Path) -> list[Path]:
    """Discover supported input images deterministically.

    Args:
        source: A single image or a directory searched recursively.

    Returns:
        Sorted absolute JPG, JPEG, and BMP paths.

    Raises:
        FileNotFoundError: If the source does not exist.
        ValueError: If a file has an unsupported format or no images are found.
    """
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {path.suffix}; expected JPG or BMP")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    images = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No JPG or BMP images found under {path}")
    return images


def _clean(value: object) -> object:
    """Convert non-finite CSV values to empty cells."""
    return "" if isinstance(value, float) and not math.isfinite(value) else value


class AnalysisPipeline:
    """Execute the three-block analysis and write specification outputs."""

    def __init__(self, config: ModuleConfig, backend: InferenceBackend | None = None) -> None:
        """Initialize the pipeline.

        Args:
            config: Module and inference configuration.
            backend: Optional backend override, primarily for integration testing.
        """
        self.config = config
        self.backend = backend or ConfigurableBackend(config)

    def validate_model_classes(self) -> list[str]:
        """Check configured classes against classes exposed by loaded models.

        Returns:
            Human-readable compatibility errors; an empty list means success.
        """
        if not hasattr(self.backend, "model_classes"):
            return []
        floc_classes, micro_classes = self.backend.model_classes()  # type: ignore[attr-defined]
        errors = []
        if self.config.floc_class_name not in floc_classes:
            errors.append(f"floc model lacks class {self.config.floc_class_name!r}; has {sorted(floc_classes)}")
        missing = set(self.config.microorganism_classes) - micro_classes
        if missing:
            errors.append(f"microorganism model lacks configured classes: {sorted(missing)}")
        return errors

    def run(self, source: str | Path, output_dir: str | Path) -> dict[str, object]:
        """Analyze all input images and persist output artifacts.

        Args:
            source: Input JPG/BMP file or directory.
            output_dir: Directory that receives CSV and JSON artifacts.

        Returns:
            Run summary with processed-image and detection counts.

        Raises:
            FileNotFoundError: If the input source does not exist.
            ValueError: If inputs cannot be decoded or contain no supported images.
        """
        images = discover_images(source)
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        floc_raw: list[dict[str, object]] = []
        detection_documents: list[dict[str, object]] = []
        count_rows: list[dict[str, object]] = []
        combined_rows: list[dict[str, object]] = []

        for image_path in images:
            image_id = image_path.stem
            image = read_image(image_path)
            segments = self.backend.segment(image)
            image_floc_rows = [
                instance_row(image_id, index, extract_features(image, segment.mask))
                for index, segment in enumerate(segments, start=1)
            ]
            floc_raw.extend(image_floc_rows or [{"image_id": image_id, "row_type": "_image"}])
            detections = self.backend.detect(image)
            counts = Counter(d.class_name for d in detections)
            captured_at = datetime.fromtimestamp(image_path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
            detection_documents.append({
                "image_id": image_id,
                "file_name": image_path.name,
                "objects": [
                    {"class": d.class_name, "bbox_xyxy": [round(v, 2) for v in d.bbox_xyxy], "confidence": round(d.confidence, 6)}
                    for d in detections
                ],
            })
            count_row: dict[str, object] = {"image_id": image_id}
            count_row.update({name: counts[name] for name in self.config.microorganism_classes})
            count_rows.append(count_row)
            if image_floc_rows:
                means = {name: float(np.nanmean([float(row[name]) for row in image_floc_rows])) for name in FEATURE_COLUMNS}
            else:
                means = {name: math.nan for name in FEATURE_COLUMNS}
            combined: dict[str, object] = {
                "image_id": image_id,
                "captured_at": captured_at,
                "floc_count": len(image_floc_rows),
                "mean_floc_area_px": means["area_px"],
                "mean_floc_compactness": means["compactness"],
                "mean_floc_fractal_dimension": means["fractal_dimension"],
                "mean_floc_brightness": means["mean_brightness"],
            }
            combined.update({name: counts[name] for name in self.config.microorganism_classes})
            combined_rows.append(combined)

        floc_rows = aggregate_rows(floc_raw)
        self._write_csv(output / "floc_features.csv", FLOC_CSV_COLUMNS, floc_rows)
        micro_fields = ("image_id", *self.config.microorganism_classes)
        self._write_csv(output / "microorganism_counts.csv", micro_fields, count_rows)
        combined_fields = (
            "image_id", "captured_at", "floc_count", "mean_floc_area_px",
            "mean_floc_compactness", "mean_floc_fractal_dimension", "mean_floc_brightness",
            *self.config.microorganism_classes,
        )
        self._write_csv(output / "combined_analysis.csv", combined_fields, combined_rows)
        (output / "microorganism_detections.json").write_text(
            json.dumps(detection_documents, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = {
            "images_processed": len(images),
            "flocs_detected": sum(int(row["floc_count"]) for row in combined_rows),
            "microorganisms_detected": sum(len(doc["objects"]) for doc in detection_documents),
            "outputs": ["floc_features.csv", "microorganism_detections.json", "microorganism_counts.csv", "combined_analysis.csv"],
        }
        (output / "run_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    @staticmethod
    def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
        """Serialize rows as an Excel-compatible UTF-8 CSV file."""
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows({key: _clean(value) for key, value in row.items()} for row in rows)


def analyze(config: str | Path | ModuleConfig, source: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Load both models and analyze a file or directory.

    Args:
        config: Parsed configuration or path to a YAML configuration file.
        source: Input JPG/BMP file or directory.
        output_dir: Directory that receives analysis artifacts.

    Returns:
        Run summary with processed-image and detection counts.
    """
    resolved = load_config(config) if not isinstance(config, ModuleConfig) else config
    return AnalysisPipeline(resolved).run(source, output_dir)

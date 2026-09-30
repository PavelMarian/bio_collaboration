"""Shared fixtures: real configuration, synthetic images and a synthetic project."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

import sludge_micro
from sludge_micro.config import ProjectConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "microorganisms.yaml"
REINSTALL = ".venv/bin/pip install --no-deps --no-build-isolation ."


def pytest_sessionstart(session: pytest.Session) -> None:
    """Stop when the installed package differs from the sources under test."""
    installed = Path(sludge_micro.__file__).parent
    source = ROOT / "src" / "sludge_micro"
    stale = sorted(
        p.name
        for p in source.glob("*.py")
        if not (installed / p.name).is_file() or (installed / p.name).read_bytes() != p.read_bytes()
    )
    if stale:
        pytest.exit(f"installed package is stale ({stale}); run {REINSTALL}", returncode=3)


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    return load_config(CONFIG)


def synthetic_image(seed: int, width: int = 64, height: int = 48) -> np.ndarray:
    """Return a textured BGR image that differs strongly between seeds."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def encode(image: np.ndarray, suffix: str) -> bytes:
    ok, data = cv2.imencode(suffix, image)
    assert ok
    return data.tobytes()


def coco_document(config: ProjectConfig, images: list[dict], annotations: list[dict]) -> dict:
    """Build a COCO document with the microorganism category schema of the archives."""
    ids = {c.source_category: c.source_category_id for c in config.classes.target}
    extra = {
        "floc": 1,
        "debris": 6,
        "suctoria": 8,
        "naked_amoebae": 10,
        "flagellates": 11,
        "fungi": 12,
        "aeolosoma": 13,
        "unknown": 15,
        "custom_microorganism": 16,
    }
    categories = [
        {"id": i, "name": n} for n, i in sorted({**ids, **extra}.items(), key=lambda kv: kv[1])
    ]
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info": {},
        "licenses": [],
    }


def write_coco_zip(path: Path, document: dict, files: dict[str, bytes], folder: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"annotations/instances_{folder}.json", json.dumps(document))
        for name, data in files.items():
            archive.writestr(f"images/{folder}/{name}", data)


def box_ann(ann_id: int, image_id: int, category_id: int, box: list[float]) -> dict:
    x, y, w, h = box
    polygon = [x, y, x + w, y, x + w, y + h, x, y + h]
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": box,
        "segmentation": [polygon],
        "area": w * h,
        "iscrowd": 0,
    }


def jpeg_with_exif(image: np.ndarray, moment: str) -> bytes:
    """Encode a JPEG that carries EXIF DateTimeOriginal."""
    exif = Image.Exif()
    exif[0x0110] = "TestCam"
    exif.get_ifd(0x8769)[0x9003] = moment
    buffer = io.BytesIO()
    Image.fromarray(image[:, :, ::-1]).save(buffer, "JPEG", exif=exif, quality=95)
    return buffer.getvalue()


def project_config(config: ProjectConfig) -> dict:
    data = config.model_dump(mode="json", exclude={"root"})
    data["paths"]["raw_dir"] = "data/raw"
    data["sources"] = [
        {
            "id": "arch_a",
            "archive": "A.zip",
            "format": "coco",
            "role": "microorganisms",
            "priority": 0,
        },
        {
            "id": "arch_b",
            "archive": "B.zip",
            "format": "coco",
            "role": "microorganisms",
            "priority": 1,
        },
    ]
    data["audit"]["expected_width"], data["audit"]["expected_height"] = 64, 48
    data["duplicate_variants"] = {
        "exclude_conflicts": None,
        "prefer_a": "arch_a",
        "prefer_b": "arch_b",
    }
    return data


def entry(image_id: int, name: str) -> dict:
    return {"id": image_id, "file_name": name, "width": 64, "height": 48, "date_captured": 0}


def project_archives(config: ProjectConfig, raw: Path) -> None:
    """A: shared copy, invalid box, dated frame with ignore region, floc-only frame; B: copy."""
    rot, att, unknown, floc = 3, 2, 15, 1
    shared = encode(synthetic_image(1), ".bmp")
    doc_a = coco_document(
        config,
        [
            entry(1, "a/one.bmp"),
            entry(2, "a/two.bmp"),
            entry(3, "a/three.jpg"),
            entry(4, "a/four.bmp"),
        ],
        [
            box_ann(1, 1, rot, [4, 4, 10, 10]),
            box_ann(2, 2, att, [60, 40, 10, 10]),
            box_ann(3, 3, rot, [10, 10, 20, 20]),
            box_ann(4, 3, unknown, [40, 5, 10, 10]),
            box_ann(5, 4, floc, [5, 5, 30, 30]),
        ],
    )
    write_coco_zip(
        raw / "A.zip",
        doc_a,
        {
            "a/one.bmp": shared,
            "a/two.bmp": encode(synthetic_image(2), ".bmp"),
            "a/three.jpg": jpeg_with_exif(synthetic_image(3), "2024:01:31 10:00:00"),
            "a/four.bmp": encode(synthetic_image(4), ".bmp"),
        },
        "Train",
    )
    doc_b = coco_document(config, [entry(1, "b/one.bmp")], [box_ann(1, 1, att, [4, 4, 10, 10])])
    write_coco_zip(raw / "B.zip", doc_b, {"b/one.bmp": shared}, "default")


@pytest.fixture
def project(tmp_path: Path, config: ProjectConfig) -> Path:
    """Create a small project with two COCO archives and a pixel-identical copy."""
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'synthetic'\n")
    (root / "experiments").mkdir()
    (root / "data" / "raw").mkdir(parents=True)
    project_archives(config, root / "data" / "raw")
    text = yaml.safe_dump(project_config(config), allow_unicode=True)
    (root / "configs" / "microorganisms.yaml").write_text(text)
    return root


def zip_bytes(entries: dict[str, bytes]) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    buffer.seek(0)
    return buffer


class Helpers:
    """Synthetic data builders shared by tests through the ``helpers`` fixture."""

    synthetic_image = staticmethod(synthetic_image)
    encode = staticmethod(encode)
    coco_document = staticmethod(coco_document)
    write_coco_zip = staticmethod(write_coco_zip)
    box_ann = staticmethod(box_ann)
    zip_bytes = staticmethod(zip_bytes)
    project_config = staticmethod(project_config)


@pytest.fixture(scope="session")
def helpers() -> type[Helpers]:
    return Helpers


class FakeBoxes:
    """Box container with the attributes read by the Ultralytics adapters."""

    def __init__(self, rows: list[list[float]]) -> None:
        import torch

        table = torch.tensor(rows, dtype=torch.float32).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = table[:, :4], table[:, 4], table[:, 5]


class FakeResult:
    def __init__(self, names: dict, rows: list[list[float]]) -> None:
        self.names, self.boxes, self.masks = names, FakeBoxes(rows), None


class FakeYOLO:
    """Test double of ``ultralytics.YOLO``; the weights file is JSON with fixed outputs."""

    calls: ClassVar[list] = []

    def __init__(self, path: str) -> None:
        document = json.loads(Path(path).read_text())
        self.path = path
        self.names = {int(k): v for k, v in document["names"].items()}
        self.rows = document["detections"]

    def predict(self, source, **kwargs):
        FakeYOLO.calls.append(("predict", kwargs))
        rows = [r for r in self.rows if r[4] >= kwargs.get("conf", 0.0)]
        return [FakeResult(self.names, rows[: kwargs.get("max_det", 300)])]

    def train(self, **kwargs):
        FakeYOLO.calls.append(("train", kwargs))
        weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "last.pt").write_text(Path(self.path).read_text())


@pytest.fixture
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> type[FakeYOLO]:
    """Install the ``ultralytics`` test double for the duration of one test."""
    import sys
    import types

    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    FakeYOLO.calls = []
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return FakeYOLO


def fake_weights(path: Path, names: list[str], detections: list[list[float]]) -> Path:
    """Write a JSON weights file for ``FakeYOLO``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"names": dict(enumerate(names)), "detections": detections}))
    return path


Helpers.fake_weights = staticmethod(fake_weights)

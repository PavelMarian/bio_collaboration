"""Delivery of a trained run: its passport from the run reports and its README summary.

``build_passport`` reads the configuration snapshot, manifest and validation
metrics of a run, its single test evaluation when there is one, its pipeline
hand-over check when there is one and the split lock, and returns the passport
of the run's weights. ``render_summary`` turns a passport into the Markdown
block that README files show between ``<!-- generated:passport -->`` markers,
so the numbers in the README always come from the passport.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sludge_micro.archives import sha256_file
from sludge_micro.config import ProjectConfig, load_config
from sludge_micro.messages import ProjectError
from sludge_micro.passport import FORMAT, WEIGHTS_NAMES, ModelPassport
from sludge_micro.reporting import read_json
from sludge_micro.runs import checkpoint_of, run_dirs, test_usage_path
from sludge_micro.split import lock_path

STATUS = "research prototype"
CLASS_KEYS = ("name", "support", "predictions", "tp", "precision", "recall", "f1")
COUNT_KEYS = ("name", "annotated", "counted", "mae_per_frame", "bias_per_frame")
START, END = "<!-- generated:passport -->", "<!-- /generated:passport -->"
BLOCK = re.compile(re.escape(START) + ".*?" + re.escape(END), re.S)


def result(metrics: dict[str, Any]) -> dict[str, Any]:
    """Conditions, macro F1 and per-class results of one evaluation."""
    out = {k: metrics[k] for k in ("images", "objects", "ignore_regions", "confidence")}
    out |= {"macro_f1": metrics["macro_f1"], "macro_f1_classes": metrics["macro_f1_classes"]}
    out["classes"] = [{k: c[k] for k in CLASS_KEYS} for c in metrics["classes"]]
    if "counting" in metrics:
        out["counting"] = [{k: c[k] for k in COUNT_KEYS} for c in metrics["counting"]["classes"]]
    return out


def evaluation(config: ProjectConfig, run_id: str, report_dir: Path) -> dict[str, Any]:
    """Evaluation rules, validation results and the single test evaluation of a run."""
    ev = config.evaluation
    out: dict[str, Any] = {
        "rules": {
            "iou_threshold": ev.iou_threshold,
            "ignore_ioa_threshold": ev.ignore_ioa_threshold,
            "matching": ev.matching,
            "unsupported_class_policy": ev.unsupported_class_policy,
            "threshold_selection": "validation grid",
        },
        "validation": result(read_json(report_dir / "metrics.json")),
        "test": None,
    }
    usage = read_json(test_usage_path(config)) if test_usage_path(config).is_file() else None
    if usage is not None and usage["run_id"] == run_id:
        out["test"] = result(read_json(config.resolve(usage["metrics"]))) | {"single_use": True}
    return out


def training(config: ProjectConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    """Training settings, initial weights and data of a run."""
    t, splits = config.train, read_json(lock_path(config))
    initial = manifest["inputs"]["pretrained_weights"] or {}
    return {
        "epochs": len(manifest["train_history"]),
        "optimizer": {
            "name": "SGD",
            "lr": t.lr,
            "momentum": t.momentum,
            "weight_decay": t.weight_decay,
        },
        "batch_size": t.batch_size,
        "accumulate": t.accumulate,
        "batchnorm": t.batchnorm,
        "trainable_backbone_layers": config.model.trainable_backbone_layers,
        "augment": t.augment.model_dump(mode="json"),
        "sampling": t.sampling,
        "seed": config.runtime.seed,
        "device": config.runtime.device,
        "initial_weights": {
            "file": Path(initial.get("path", "")).name or None,
            "sha256": initial.get("sha256"),
        },
        "data_variant": config.prepare.variant,
        "split": {
            "protocol": config.split.protocol,
            "manifest_sha256": splits["manifest_sha256"],
            "train_images_after_filter": manifest["train_images"],
            "validation_images": splits["splits"]["validation"]["images"],
            "test_images": splits["splits"]["test"]["images"],
        },
    }


def integration(report_dir: Path) -> dict[str, Any] | None:
    """Pipeline hand-over check of the run, when it was made."""
    path = report_dir / "pavel_handover.json"
    if not path.is_file():
        return None
    h = read_json(path)
    keys = ("images", "objects", "detections_match", "counts_match_detections")
    return {k: h[k] for k in keys} | {"class_errors": len(h["class_errors"])}


def image_section(run: ProjectConfig) -> dict[str, Any]:
    """Image formats, colour handling and resize bounds of a run."""
    return {
        "formats": list(run.audit.image_suffixes),
        "color": "opencv_bgr_uint8_to_rgb_float",
        "value_range": [0.0, 1.0],
        "expected_width": run.audit.expected_width,
        "expected_height": run.audit.expected_height,
        "min_size": run.model.min_size,
        "max_size": run.model.max_size,
    }


def detector_section(run: ProjectConfig, confidence: float) -> dict[str, Any]:
    """Architecture, post-processing and the validation threshold of a run."""
    post = run.postprocess
    return {
        "architecture": run.model.architecture,
        "min_score": post.min_score,
        "nms_iou": post.nms_iou,
        "max_detections": post.max_detections,
        "confidence": confidence,
    }


def weights_section(run: ProjectConfig, weights: Path) -> dict[str, Any]:
    """Delivered file name, checksum, size and checkpoint format of a run's weights."""
    return {
        "file": WEIGHTS_NAMES[run.model.backend],
        "sha256": sha256_file(weights),
        "bytes": weights.stat().st_size,
        "checkpoint_format": read_checkpoint_format(weights),
    }


def build_passport(config: ProjectConfig, run_id: str) -> dict[str, Any]:
    """Passport of a run's weights from its reports; the run's own snapshot sets the values.

    Raises:
        ProjectError: If the run has no validation metrics, manifest or snapshot.
    """
    report_dir = run_dirs(config, run_id)[0]
    for name in ("metrics.json", "manifest.json", "config.yaml"):
        if not (report_dir / name).is_file():
            raise ProjectError("run_missing", run_id=run_id, path=report_dir / name)
    run = load_config(report_dir / "config.yaml")
    manifest = read_json(report_dir / "manifest.json")
    confidence = read_json(report_dir / "metrics.json")["confidence"]
    passport = {
        "format": FORMAT,
        "run_id": run_id,
        "status": STATUS,
        "classes": [
            {"label": c.label, "name": c.name, "title": c.title} for c in run.classes.target
        ],
        "image": image_section(run),
        "detector": detector_section(run, confidence),
        "weights": weights_section(run, checkpoint_of(config, run_id)),
        "training": training(run, manifest),
        "evaluation": evaluation(run, run_id, report_dir),
        "integration": integration(report_dir),
    }
    return ModelPassport.model_validate(passport).model_dump(mode="json")


def read_checkpoint_format(path: Path) -> str:
    """Format tag of an own checkpoint."""
    from sludge_micro.checkpoint import load_restricted

    return str(load_restricted(path).get("format"))


def num(value: float | int | None, digits: int = 3) -> str:
    """Number for Russian text with a decimal comma."""
    if value is None:
        return "нет данных"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}".replace(".", ",")


def table(header: list[str], rows: list[list[str]]) -> str:
    """Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def settings_rows(p: ModelPassport) -> list[list[str]]:
    """Architecture, image handling, post-processing, threshold and weights of a passport."""
    i, d, w = p.image, p.detector, p.weights
    return [
        ["Архитектура", f"`{d.architecture}`"],
        [
            "Вход",
            f"{', '.join(i.formats)}; ожидаемый размер {i.expected_width} x {i.expected_height}",
        ],
        ["Цвет и диапазон", "OpenCV BGR `uint8` переводится в RGB `float32` от 0 до 1"],
        ["Масштабирование", f"короткая сторона {i.min_size}, длинная не больше {i.max_size}"],
        [
            "Постобработка детектора",
            f"оценка от {num(d.min_score, 2)}, NMS IoU {num(d.nms_iou, 2)}, "
            f"не больше {d.max_detections} детекций",
        ],
        ["Порог confidence", f"{num(d.confidence, 2)}, выбран на validation"],
        ["Веса", f"`{w.file}`, {w.bytes} байт, SHA-256 `{w.sha256}`"],
    ]


def result_rows(p: ModelPassport) -> list[list[str]]:
    """Validation and test conditions and macro F1."""
    rows = []
    for name, label in (("validation", "validation"), ("test", "test, однократная оценка")):
        r = p.evaluation.get(name)
        if r:
            rows.append(
                [
                    label,
                    num(r["images"]),
                    num(r["objects"]),
                    num(r["ignore_regions"]),
                    num(r["confidence"], 2),
                    num(r["macro_f1"]),
                ]
            )
    return rows


def class_rows(p: ModelPassport) -> list[list[str]]:
    """Per-class validation and test results and test count errors."""
    val = {c["name"]: c for c in p.evaluation["validation"]["classes"]}
    test = p.evaluation.get("test") or {}
    t = {c["name"]: c for c in test.get("classes", [])}
    counts = {c["name"]: c for c in test.get("counting", [])}
    rows = []
    for c in p.classes:
        v, x, k = val[c.name], t.get(c.name), counts.get(c.name)
        rows.append(
            [
                str(c.label),
                f"`{c.name}`",
                c.title,
                f"{v['support']}: {num(v['precision'])} / {num(v['recall'])} / {num(v['f1'])}",
                (
                    f"{x['support']}: {num(x['precision'])} / {num(x['recall'])} / {num(x['f1'])}"
                    if x
                    else "нет"
                ),
                f"{num(k['mae_per_frame'], 2)} / {num(k['bias_per_frame'], 2)}" if k else "нет",
            ]
        )
    return rows


def render_summary(p: ModelPassport) -> str:
    """Markdown summary of a passport for the README block."""
    rules = p.evaluation["rules"]
    integration_text = "не проверялась"
    if p.integration:
        g = p.integration
        integration_text = (
            f"кадров {g['images']}, объектов {g['objects']}; детекции пайплайна совпали с "
            f"собственным инференсом на {g['detections_match']} кадрах, счётчики совпали с "
            f"детекциями на {g['counts_match_detections']} кадрах, "
            f"ошибок классов {g['class_errors']}"
        )
    return "\n\n".join(
        [
            f"Паспорт `model/passport.json`: запуск `{p.run_id}`, "
            "статус: исследовательский прототип.",
            table(["Параметр", "Значение"], settings_rows(p)),
            f"Качество: F1 считается по объектам при IoU не ниже {num(rules['iou_threshold'], 2)}, "
            "macro F1 равен среднему F1 по классам. Это не доля верно распознанных снимков.",
            table(
                [
                    "Выборка",
                    "Изображений",
                    "Объектов",
                    "Игнорируемых областей",
                    "Порог",
                    "macro F1",
                ],
                result_rows(p),
            ),
            table(
                [
                    "Label",
                    "Класс",
                    "Название",
                    "Validation: объектов: P / R / F1",
                    "Test: объектов: P / R / F1",
                    "Test: ошибка подсчёта на кадр, средняя абсолютная / смещение",
                ],
                class_rows(p),
            ),
            f"Проверка передачи в пайплайн Павла на кадрах validation: {integration_text}.",
        ]
    )


def update_readme(readme: Path, passport: ModelPassport) -> bool:
    """Rewrite the generated passport block of a README; return whether it changed.

    Raises:
        ProjectError: If the README has no passport block.
    """
    text = readme.read_text(encoding="utf-8")
    if not BLOCK.search(text):
        raise ProjectError("readme_block_missing", path=readme)
    block = f"{START}\n\n{render_summary(passport)}\n\n{END}"
    fresh = BLOCK.sub(lambda _: block, text)
    if fresh != text:
        readme.write_text(fresh, encoding="utf-8")
    return fresh != text

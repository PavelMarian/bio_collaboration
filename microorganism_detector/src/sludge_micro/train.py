"""Training of one configured run.

Preconditions are checked before any computation: a locked split, local
pretrained weights and the run budget. Wall-clock durations go to the
technical log ``outputs/<run_id>/train_log.jsonl`` and never to reports.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN

from sludge_micro.config import ProjectConfig
from sludge_micro.dataset import Sample, class_presence, make_loader
from sludge_micro.messages import ProjectError
from sludge_micro.model import (
    build_detector,
    freeze_backbone,
    init_from_pretrained,
    save_checkpoint,
    torch_device,
)
from sludge_micro.reporting import read_json, relative
from sludge_micro.runs import evaluate_run, run_dirs, split_samples, write_run_manifest
from sludge_micro.runtime import apply_runtime, code_at_start
from sludge_micro.split import load_split, lock_path

LOG = logging.getLogger(__name__)
# Technical progress line interval; it does not affect training.
PROGRESS_EVERY = 25
RESUME_FILE = "resume_state.pt"
BATCHNORM = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def check_budget(config: ProjectConfig, run_id: str) -> None:
    """Refuse a new run when the configured number of runs is spent."""
    done = {
        p.parent.name for p in config.resolve(config.paths.experiments_dir).glob("*/metrics.json")
    } - {run_id}
    if len(done) >= config.protocol.max_runs:
        raise ProjectError("budget_exhausted", runs=len(done), limit=config.protocol.max_runs)


def pretrained_path(config: ProjectConfig) -> Path | None:
    """Return the configured pretrained weights path, if any."""
    weights = config.paths.pretrained_weights
    return config.resolve(weights) if weights is not None else None


def freeze_batchnorm(model: nn.Module) -> int:
    """Keep every BatchNorm layer on its stored statistics while the model trains."""
    layers = [m for m in model.modules() if isinstance(m, BATCHNORM)]
    for layer in layers:
        layer.eval()
    return len(layers)


def set_training_mode(model: nn.Module, batchnorm: str) -> None:
    """Switch the model to training; ``frozen`` then returns BatchNorm layers to eval mode."""
    model.train()
    if batchnorm == "frozen":
        freeze_batchnorm(model)


def batchnorm_statistics(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copies of the stored statistics of every BatchNorm layer, keyed by buffer name."""
    return {
        f"{name}.{buffer}": getattr(module, buffer).detach().cpu().clone()
        for name, module in model.named_modules()
        if isinstance(module, BATCHNORM)
        for buffer in ("running_mean", "running_var", "num_batches_tracked")
    }


def batchnorm_check(
    model: nn.Module, start: dict[str, torch.Tensor], batchnorm: str
) -> dict[str, Any]:
    """Compare the stored BatchNorm statistics of a model with those at the start of a run."""
    now = batchnorm_statistics(model)
    changed = sorted(k for k in start if not torch.equal(start[k], now[k]))
    layers = [m for m in model.modules() if isinstance(m, BATCHNORM)]
    return {
        "mode": batchnorm,
        "layers": len(layers),
        "layers_in_training_mode": sum(m.training for m in layers),
        "statistics": len(start),
        "changed": len(changed),
        "changed_first": changed[:5],
    }


def sampling_record(samples: list[Sample], config: ProjectConfig) -> dict[str, Any]:
    """Sampling policy and the class presence of the training images its weights use."""
    presence = class_presence(samples)
    names = config.classes.names
    return {
        "policy": config.train.sampling,
        "images": len(samples),
        "presence": {name: presence.get(label, 0) for label, name in enumerate(names, start=1)},
    }


def trainable_parameters(model: nn.Module) -> dict[str, Any]:
    """Names and sizes of the parameters that the optimizer updates."""
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    return {
        "tensors": len(names),
        "values": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "names_sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
    }


def train_epoch(
    model: FasterRCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accumulate: int,
    device: torch.device,
    batchnorm: str = "batch",
) -> float:
    """Run one epoch with gradient accumulation and return the mean loss.

    Args:
        batchnorm: ``frozen`` keeps BatchNorm layers on stored statistics, ``batch``
            normalises with the statistics of each training batch.
    """
    set_training_mode(model, batchnorm)
    optimizer.zero_grad()
    total, steps = 0.0, 0
    for steps, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = sum(model(images, targets).values())
        (loss / accumulate).backward()
        if steps % accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
        total += float(loss.detach())
        if steps % PROGRESS_EVERY == 0:
            LOG.info("step %d/%d loss %.4f", steps, len(loader), total / steps)
    if steps % accumulate:
        optimizer.step()
        optimizer.zero_grad()
    return total / max(steps, 1)


def save_resume_state(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    history: list[dict[str, Any]],
) -> None:
    """Write everything that an exact continuation needs, atomically."""
    state = {
        "epoch": len(history),
        "history": history,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "generator": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    temporary = path.with_name(path.name + ".part")
    torch.save(state, temporary)
    os.replace(temporary, path)


def restore_resume_state(
    state: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> list[dict[str, Any]]:
    """Restore model, optimizer and all random generators; return the loss history."""
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    generator.set_state(state["generator"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return list(state["history"])


def fit(
    model: nn.Module,
    loader: DataLoader,
    config: ProjectConfig,
    device: torch.device,
    paths: tuple[Path, Path],
    generator: torch.Generator,
    resume: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Train for the configured epochs, saving a resume state after every epoch.

    Args:
        paths: Technical log and resume state file.
        resume: Saved state to continue from, or ``None`` for a new run.
    """
    log, state_path = paths
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.train.lr,
        momentum=config.train.momentum,
        weight_decay=config.train.weight_decay,
    )
    history = restore_resume_state(resume, model, optimizer, generator) if resume else []
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        for epoch in range(len(history), config.train.epochs):
            started = time.perf_counter()
            loader.dataset.set_epoch(epoch)
            loss = train_epoch(
                model, loader, optimizer, config.train.accumulate, device, config.train.batchnorm
            )
            history.append({"epoch": epoch + 1, "loss": loss})
            save_resume_state(state_path, model, optimizer, generator, history)
            seconds = time.perf_counter() - started
            handle.write(json.dumps({"epoch": epoch + 1, "loss": loss, "seconds": seconds}) + "\n")
            handle.flush()
            LOG.info("epoch %d loss %.4f", epoch + 1, loss)
    return history


def build_for_training(config: ProjectConfig) -> tuple[FasterRCNN, dict[str, Any]]:
    """Build the detector, load local COCO weights and freeze early stages."""
    model = build_detector(config)
    info = init_from_pretrained(model, pretrained_path(config), config.model.architecture)
    freeze_backbone(model, config.model.trainable_backbone_layers)
    return model, info


def resume_state_for(config: ProjectConfig, run_id: str, resume: bool) -> dict[str, Any] | None:
    """Load the saved state for ``--resume`` or refuse to overwrite an existing run."""
    report_dir, out_dir = run_dirs(config, run_id)
    state_path = out_dir / RESUME_FILE
    if resume:
        if not state_path.is_file():
            raise ProjectError("resume_missing", path=relative(state_path, config.root))
        return torch.load(state_path, map_location="cpu", weights_only=False)
    if (
        (out_dir / "model.pt").exists()
        or (report_dir / "metrics.json").exists()
        or state_path.exists()
    ):
        raise ProjectError("run_exists", run_id=run_id, path=relative(out_dir, config.root))
    return None


def check_preconditions(config: ProjectConfig, run_id: str) -> None:
    """Refuse a run without a locked split, budget or local initial weights."""
    load_split(config)
    check_budget(config, run_id)
    pretrained = pretrained_path(config)
    if pretrained is None or not pretrained.is_file():
        raise ProjectError(
            "pretrained_missing", path=pretrained, architecture=config.model.architecture
        )


def train_model(
    config: ProjectConfig,
    run_id: str,
    state: dict[str, Any] | None,
    generator: torch.Generator,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the detector from COCO weights, train it and check its BatchNorm statistics.

    The statistics at the start of the run are those of the initial weights; a
    continued run is checked against the same start.

    Returns:
        The trained model and the facts recorded in the run manifest.
    """
    out_dir = run_dirs(config, run_id)[1]
    samples, _ = split_samples(config, "train")
    model, init_info = build_for_training(config)
    start = batchnorm_statistics(model)
    trainable = trainable_parameters(model)
    device = torch_device(config.runtime.device)
    model.to(device)
    loader = make_loader(samples, config, True, generator)
    paths = (out_dir / "train_log.jsonl", out_dir / RESUME_FILE)
    history = fit(model, loader, config, device, paths, generator, state)
    set_training_mode(model, config.train.batchnorm)
    return model, {
        "train_history": history,
        "initialisation": init_info,
        "train_images": len(samples),
        "resumed_after_epoch": state["epoch"] if state else None,
        "batchnorm": batchnorm_check(model, start, config.train.batchnorm),
        "trainable_parameters": trainable,
        "sampling": sampling_record(samples, config),
    }


def run_train(config: ProjectConfig, run_id: str, resume: bool = False) -> dict[str, Any]:
    """Train a run, save its checkpoint and evaluate it on validation.

    With ``resume`` the run continues after the last completed epoch from the
    saved model, optimizer and random-generator state.

    Returns:
        Validation metrics of the run.
    """
    if config.model.backend == "ultralytics":
        from sludge_micro.yolo import run_train_yolo

        return run_train_yolo(config, run_id, resume)
    started = code_at_start(config.root)
    check_preconditions(config, run_id)
    state = resume_state_for(config, run_id, resume)
    generator = apply_runtime(config.runtime)
    report_dir, out_dir = run_dirs(config, run_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    model, facts = train_model(config, run_id, state, generator)
    lock = read_json(lock_path(config))
    save_checkpoint(
        model,
        out_dir / "model.pt",
        config,
        {"run_id": run_id, "split_manifest_sha256": lock["manifest_sha256"]},
    )
    metrics = evaluate_run(config, run_id, "validation")
    write_run_manifest(config, run_id, {"code_at_start": started, **facts})
    return metrics


def hardware() -> dict[str, Any]:
    """Machine facts that bound training time and memory."""
    import os
    import platform
    import subprocess

    def sysctl(name: str) -> str:
        result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False)
        return result.stdout.strip()

    return {
        "machine": platform.machine(),
        "cpu": sysctl("machdep.cpu.brand_string") or None,
        "logical_cpus": os.cpu_count(),
        "memory_bytes": int(sysctl("hw.memsize") or 0) or None,
        "mps_available": torch.backends.mps.is_available(),
    }


def torchvision_step(config: ProjectConfig) -> dict[str, Any]:
    """Load weights and run one training step on the first training image."""
    generator = apply_runtime(config.runtime)
    samples, _ = split_samples(config, "train")
    model, info = build_for_training(config)
    start = batchnorm_statistics(model)
    device = torch_device(config.runtime.device)
    model.to(device)
    set_training_mode(model, config.train.batchnorm)
    loader = make_loader(samples[:1], config, True, generator)
    images, targets = next(iter(loader))
    started = time.perf_counter()
    losses = model(
        [i.to(device) for i in images], [{k: v.to(device) for k, v in t.items()} for t in targets]
    )
    loss = sum(losses.values())
    loss.backward()
    return {
        "initialisation": info,
        "train_images": len(samples),
        "loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "step_seconds": time.perf_counter() - started,
        "classifier_outputs": len(config.classes.names) + 1,
        "batchnorm_after_step": batchnorm_check(model, start, config.train.batchnorm),
        "trainable_parameters": trainable_parameters(model),
    }


def run_preflight(config: ProjectConfig) -> dict[str, Any]:
    """Check hardware, weights, class order and one training step before a run."""
    report: dict[str, Any] = {
        "architecture": config.model.architecture,
        "hardware": hardware(),
        "device": config.runtime.device,
        "threads": config.runtime.threads,
    }
    if config.model.backend == "ultralytics":
        from sludge_micro.yolo import preflight_yolo

        report |= preflight_yolo(config)
    else:
        report |= torchvision_step(config)
    return report

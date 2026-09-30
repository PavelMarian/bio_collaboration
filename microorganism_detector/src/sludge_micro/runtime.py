"""Seeds, threads, network guard and reproducibility metadata."""

from __future__ import annotations

import os
import platform
import random
import socket
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sludge_micro.config import RuntimeConfig
from sludge_micro.messages import ProjectError

TRACKED_PACKAGES = (
    "numpy",
    "opencv-python-headless",
    "pandas",
    "pillow",
    "pyarrow",
    "pydantic",
    "PyYAML",
    "torch",
    "torchvision",
    "ultralytics",
)


def apply_runtime(runtime: RuntimeConfig) -> torch.Generator:
    """Seed every generator, fix threads and return the torch generator.

    The same seed feeds ``random``, NumPy, torch and data-loader workers.
    """
    random.seed(runtime.seed)
    np.random.seed(runtime.seed % 2**32)
    torch.manual_seed(runtime.seed)
    torch.set_num_threads(runtime.threads)
    os.environ["OMP_NUM_THREADS"] = str(runtime.threads)
    if runtime.deterministic:
        torch.use_deterministic_algorithms(True)
    generator = torch.Generator()
    generator.manual_seed(runtime.seed)
    return generator


def worker_seed(worker_id: int) -> None:
    """Seed a data-loader worker from the torch initial seed."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % 2**32)


def package_versions() -> dict[str, str | None]:
    """Return installed versions of tracked packages."""
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def code_version(root: Path) -> dict[str, Any]:
    """Return the Git commit and whether tracked files differ from it."""

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    commit = git("rev-parse", "HEAD") or None
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    return {"commit": commit, "dirty": dirty}


def package_digest() -> str:
    """SHA-256 over the installed package modules that actually run."""
    import hashlib

    import sludge_micro

    folder = Path(sludge_micro.__file__).parent
    lines = "".join(
        f"{p.name}:{hashlib.sha256(p.read_bytes()).hexdigest()}\n"
        for p in sorted(folder.glob("*.py"))
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def code_at_start(root: Path) -> dict[str, Any]:
    """Git state and installed package digest captured when a run starts."""
    return code_version(root) | {"package_sha256": package_digest()}


def environment(device: str) -> dict[str, Any]:
    """Describe interpreter, platform and device."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": device,
        "torch_threads": torch.get_num_threads(),
        "packages": package_versions(),
    }


@contextmanager
def network_disabled() -> Iterator[None]:
    """Refuse outgoing socket connections inside the block."""
    original_connect = socket.socket.connect
    original_create = socket.create_connection

    def refuse(*args: Any, **_: Any) -> None:  # noqa: ANN401
        raise ProjectError("network_disabled", target=args[1:] or args)

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.create_connection = original_create

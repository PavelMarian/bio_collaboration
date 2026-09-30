"""Canonical machine-readable reports.

Canonical JSON uses sorted keys, UTF-8 and a trailing newline. It contains no
wall-clock time or duration; those go to technical logs under ``outputs/``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:  # noqa: ANN401
    """Convert numpy, pandas and path values to plain JSON values."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        items = [to_jsonable(v) for v in value]
        return sorted(items, key=str) if isinstance(value, set) else items
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def canonical_json(data: Any) -> str:  # noqa: ANN401
    """Serialise data as canonical JSON text."""
    return json.dumps(to_jsonable(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, data: Any) -> None:  # noqa: ANN401
    """Write canonical JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(data), encoding="utf-8")


def read_json(path: Path) -> Any:  # noqa: ANN401
    """Read a JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_parquet(path: Path, table: pd.DataFrame) -> None:
    """Write a table to Parquet without the pandas index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index(drop=True).to_parquet(path, index=False)


def relative(path: Path, root: Path) -> str:
    """Return a POSIX path relative to the project root when possible.

    Symbolic links are not followed, so linked inputs keep project-relative names.
    """
    absolute, base = Path(os.path.abspath(path)), Path(os.path.abspath(root))
    return absolute.relative_to(base).as_posix() if absolute.is_relative_to(base) else str(path)

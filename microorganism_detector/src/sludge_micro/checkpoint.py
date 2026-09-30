"""Restricted checkpoint loading and inspection.

The reference checkpoint stores ``pathlib.WindowsPath`` metadata. The
restricted unpickler maps it to ``PureWindowsPath`` for this load only and
refuses every class outside the allow-list, so no global path class is patched
and no arbitrary code runs.
"""

from __future__ import annotations

import collections
import pathlib
import pickle
import re
import types
from pathlib import Path
from typing import Any

import torch

from sludge_micro.archives import sha256_file
from sludge_micro.messages import ProjectError

ALLOWED_GLOBALS: dict[tuple[str, str], Any] = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
    ("torch", "FloatStorage"): torch.FloatStorage,
    ("torch", "LongStorage"): torch.LongStorage,
    ("torch", "IntStorage"): torch.IntStorage,
    ("torch", "BoolStorage"): torch.BoolStorage,
    ("pathlib", "WindowsPath"): pathlib.PureWindowsPath,
    ("pathlib", "PosixPath"): pathlib.PurePosixPath,
    ("pathlib._local", "WindowsPath"): pathlib.PureWindowsPath,
    ("pathlib._local", "PosixPath"): pathlib.PurePosixPath,
}

V2_BOX_HEAD = re.compile(r"^roi_heads\.box_head\.\d+\.\d+\.weight$")


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler limited to tensors, ordered dicts and pure paths."""

    def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401
        """Resolve only allow-listed globals."""
        try:
            return ALLOWED_GLOBALS[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"{module}.{name} is not allowed") from exc


RESTRICTED_PICKLE = types.SimpleNamespace(
    Unpickler=RestrictedUnpickler, load=pickle.load, __name__="restricted_pickle"
)


def load_restricted(path: Path) -> dict[str, Any]:
    """Load a checkpoint dictionary through the restricted unpickler.

    Raises:
        ProjectError: If the file is missing or contains disallowed objects.
    """
    if not path.is_file():
        raise ProjectError("file_missing", path=path, action="проверьте путь к весам.")
    try:
        loaded = torch.load(
            path, map_location="cpu", weights_only=False, pickle_module=RESTRICTED_PICKLE
        )
    except (pickle.UnpicklingError, RuntimeError, EOFError) as exc:
        raise ProjectError("checkpoint_unreadable", path=path, details=exc) from exc
    if not isinstance(loaded, dict):
        raise ProjectError("checkpoint_unreadable", path=path, details=type(loaded).__name__)
    return loaded


def state_dict_of(checkpoint: dict[str, Any], path: Path) -> dict[str, torch.Tensor]:
    """Return the model state dictionary stored in a checkpoint."""
    for key in ("model", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return {str(k).removeprefix("module."): v for k, v in value.items()}
    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return {str(k).removeprefix("module."): v for k, v in checkpoint.items()}
    raise ProjectError("checkpoint_unreadable", path=path, details="state dict not found")


RETINA_CLS = "head.classification_head.cls_logits.weight"
RETINA_REG = "head.regression_head.bbox_reg.weight"


def head_outputs(state: dict[str, torch.Tensor]) -> int | None:
    """Return the number of class outputs, background column included."""
    weight = state.get("roi_heads.box_predictor.cls_score.weight")
    if weight is not None:
        return int(weight.shape[0])
    if RETINA_CLS in state and RETINA_REG in state:
        anchors = int(state[RETINA_REG].shape[0]) // 4
        return int(state[RETINA_CLS].shape[0]) // anchors
    return None


def architecture_hint(state: dict[str, torch.Tensor]) -> str:
    """Describe the detector family from the parameter layout."""
    if RETINA_CLS in state:
        v2 = "head.classification_head.conv.0.1.weight" in state
        return "retinanet_resnet50_fpn_v2" if v2 else "retinanet_resnet50_fpn"
    has_mask = any(k.startswith("roi_heads.mask_") for k in state)
    v2_head = any(V2_BOX_HEAD.match(k) for k in state)
    family = "maskrcnn" if has_mask else "fasterrcnn"
    return f"{family}_resnet50_fpn_v2" if v2_head else f"{family}_resnet50_fpn"


def plain(value: Any) -> Any:  # noqa: ANN401
    """Convert checkpoint metadata to JSON-compatible values."""
    if isinstance(value, pathlib.PurePath):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, torch.Tensor):
        return f"tensor{tuple(value.shape)}"
    return value


def inspect_checkpoint(path: Path, strict_architecture: str | None) -> dict[str, Any]:
    """Describe a checkpoint without modifying it.

    Args:
        path: Checkpoint file.
        strict_architecture: Architecture to try a strict load against, if any.

    Returns:
        JSON-compatible description: hashes, keys, head size and metadata.
    """
    checkpoint = load_restricted(path)
    state = state_dict_of(checkpoint, path)
    outputs = head_outputs(state)
    info: dict[str, Any] = {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "top_level_keys": sorted(checkpoint),
        "parameter_tensors": len(state),
        "architecture_hint": architecture_hint(state),
        "classifier_outputs": outputs,
        "target_classes_implied": outputs - 1 if outputs else None,
        "metadata": {k: plain(v) for k, v in sorted(checkpoint.items()) if k != "model"},
        "class_names": plain(checkpoint.get("classes")),
    }
    if strict_architecture and outputs:
        info["strict_load"] = strict_load_report(state, strict_architecture, outputs)
    return info


def strict_load_report(state: dict[str, torch.Tensor], architecture: str, outputs: int) -> dict:
    """Try a strict load into a freshly built model without pretrained weights."""
    from sludge_micro.model import build_detector_raw

    model = build_detector_raw(architecture, outputs, None, None)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        return {"architecture": architecture, "ok": False, "error": str(exc).splitlines()[0]}
    return {"architecture": architecture, "ok": True, "error": None}

"""Restricted checkpoint loading, head checks and class-order checks."""

from __future__ import annotations

import io
import os
import pathlib
import pickle
import re

import pytest
import torch

from sludge_micro.checkpoint import (
    RestrictedUnpickler,
    architecture_hint,
    head_outputs,
    inspect_checkpoint,
    load_restricted,
)
from sludge_micro.messages import ProjectError
from sludge_micro.model import build_detector, load_trained, save_checkpoint


class WindowsPathPayload:
    """Pickles as ``pathlib.WindowsPath`` like the received checkpoint."""

    def __reduce__(self):
        return (pathlib.WindowsPath, ("C:\\weights\\coco.pth",))


class CodePayload:
    """Pickles as a call to an arbitrary function."""

    def __reduce__(self):
        return (os.getcwd, ())


def test_windows_path_loads_as_pure_path(tmp_path):
    original = pathlib.WindowsPath
    path = tmp_path / "ck.pt"
    torch.save({"model": {"w": torch.zeros(2)}, "config": {"weights": WindowsPathPayload()}}, path)
    loaded = load_restricted(path)
    assert loaded["config"]["weights"] == pathlib.PureWindowsPath("C:\\weights\\coco.pth")
    assert pathlib.WindowsPath is original


def test_legacy_pathlib_global_maps_to_pure_path():
    raw = pickle.dumps(pathlib.PureWindowsPath("C:/w/c.pth"), protocol=2)
    legacy = re.sub(rb"cpathlib[._a-z]*\nPureWindowsPath\n", b"cpathlib\nWindowsPath\n", raw)
    assert b"cpathlib\nWindowsPath\n" in legacy
    loaded = RestrictedUnpickler(io.BytesIO(legacy)).load()
    assert loaded == pathlib.PureWindowsPath("C:/w/c.pth")


def test_disallowed_globals_are_refused(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"model": {}, "config": CodePayload()}, path)
    with pytest.raises(ProjectError) as info:
        load_restricted(path)
    assert info.value.key == "checkpoint_unreadable"


def test_head_size_and_family_from_parameter_layout():
    state = {
        "roi_heads.box_predictor.cls_score.weight": torch.zeros(8, 4),
        "roi_heads.box_head.0.0.weight": torch.zeros(1),
    }
    assert head_outputs(state) == 8
    assert architecture_hint(state) == "fasterrcnn_resnet50_fpn_v2"
    assert architecture_hint(state | {"roi_heads.mask_head.0.weight": torch.zeros(1)}).startswith(
        "maskrcnn"
    )


@pytest.mark.slow
def test_own_checkpoint_round_trip_and_mismatches(tmp_path, config):
    model = build_detector(config)
    path = tmp_path / "own.pt"
    save_checkpoint(model, path, config, {"confidence": 0.5})
    _, meta = load_trained(path, config)
    assert meta["classes"] == list(config.classes.names)
    info = inspect_checkpoint(path, config.model.architecture)
    assert info["classifier_outputs"] == len(config.classes.names) + 1
    assert info["strict_load"]["ok"] is True
    payload = torch.load(path, weights_only=False)
    payload["classes"] = list(reversed(payload["classes"]))
    torch.save(payload, path)
    with pytest.raises(ProjectError) as info_err:
        load_trained(path, config)
    assert info_err.value.key == "checkpoint_classes_mismatch"
    del payload["classes"]
    torch.save(payload, path)
    with pytest.raises(ProjectError) as info_missing:
        load_trained(path, config)
    assert info_missing.value.key == "checkpoint_classes_missing"


def test_head_mismatch_is_reported(tmp_path, config):
    payload = {
        "model": {"roi_heads.box_predictor.cls_score.weight": torch.zeros(5, 4)},
        "classes": list(config.classes.names),
        "config": {"architecture": config.model.architecture},
    }
    path = tmp_path / "small.pt"
    torch.save(payload, path)
    with pytest.raises(ProjectError) as info:
        load_trained(path, config)
    assert info.value.key == "checkpoint_head_mismatch"


def test_torchvision_hash_suffix_is_verified(tmp_path):
    import hashlib

    from sludge_micro.model import check_hash_suffix

    data = b"weights"
    good = tmp_path / f"w-{hashlib.sha256(data).hexdigest()[:8]}.pth"
    good.write_bytes(data)
    check_hash_suffix(good)
    bad = tmp_path / "w-deadbeef.pth"
    bad.write_bytes(data)
    with pytest.raises(ProjectError) as info:
        check_hash_suffix(bad)
    assert info.value.key == "pretrained_hash_mismatch"
    plain = tmp_path / "coco.pth"
    plain.write_bytes(data)
    check_hash_suffix(plain)

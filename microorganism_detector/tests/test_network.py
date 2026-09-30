"""The program works without network access and never opens connections."""

from __future__ import annotations

import socket

import pytest

from sludge_micro.messages import ProjectError
from sludge_micro.model import build_detector
from sludge_micro.runtime import network_disabled


def test_connections_are_refused_inside_the_guard():
    with network_disabled(), pytest.raises(ProjectError) as info:
        socket.create_connection(("192.0.2.1", 80), timeout=0.1)
    assert info.value.key == "network_disabled"


def test_guard_restores_socket_functions():
    before = socket.socket.connect
    with network_disabled():
        pass
    assert socket.socket.connect is before


def test_detector_builds_without_network(config):
    with network_disabled():
        model = build_detector(config)
    assert model.roi_heads.box_predictor.cls_score.out_features == len(config.classes.names) + 1


def test_cli_audit_runs_offline(project):
    from sludge_micro.cli import main

    assert main(["audit", "--config", str(project / "configs" / "microorganisms.yaml")]) == 0

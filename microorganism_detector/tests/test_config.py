"""Configuration schema, class map and role assignment."""

from __future__ import annotations

import pytest
import yaml

from sludge_micro.config import load_config
from sludge_micro.messages import ProjectError

EXPECTED = (
    "attached_ciliates",
    "filamentous_bacteria",
    "rotifers",
    "testate_amoebae",
    "free_swimming_ciliates",
    "nematoda",
    "gastrotrichs",
)


def test_target_classes_follow_specification_order(config):
    assert config.classes.names == EXPECTED
    assert [c.label for c in config.classes.target] == list(range(1, len(EXPECTED) + 1))


def test_roles_follow_specification(config):
    role = config.classes.role_of
    assert role("debris") == "background"
    assert {role("unknown"), role("custom_microorganism")} == {"ignore"}
    for name in ("fungi", "flagellates", "aeolosoma", "naked_amoebae", "suctoria"):
        assert role(name) == "out_of_scope"
    assert role("floc") == "not_detected"
    assert role("nonexistent") is None


def test_label_is_not_the_source_category_id(config):
    pairs = {(c.label, c.source_category_id) for c in config.classes.target}
    assert any(label != cid for label, cid in pairs)


def test_paths_resolve_against_project_root(config):
    assert (config.root / "pyproject.toml").is_file()
    assert (config.resolve(config.paths.pavel_pipeline) / "cv_module").is_dir()


def rewrite(project, change):
    path = project / "configs" / "microorganisms.yaml"
    data = yaml.safe_load(path.read_text())
    change(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True))
    return path


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["classes"]["target"][0].update(label=5),
        lambda d: d["classes"]["background"].append("rotifers"),
        lambda d: d["evaluation"].update(iou_threshold=1.5),
        lambda d: d.update(unexpected=True),
        lambda d: d["train"].update(epochs=99),
    ],
)
def test_invalid_configuration_is_rejected(project, change):
    with pytest.raises(ProjectError) as info:
        load_config(rewrite(project, change))
    assert info.value.key == "config_invalid"


def test_configuration_is_immutable(config):
    with pytest.raises(Exception):  # noqa: B017
        config.runtime.seed = 1


def test_symlinked_inputs_keep_project_relative_names(tmp_path):
    from sludge_micro.reporting import relative

    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "w.pt").write_bytes(b"x")
    project = tmp_path / "proj"
    (project / "outputs").mkdir(parents=True)
    (project / "outputs" / "w.pt").symlink_to(tmp_path / "real" / "w.pt")
    assert relative(project / "outputs" / "w.pt", project) == "outputs/w.pt"


def test_control_dir_needs_a_control_run(config):
    from pydantic import ValidationError

    from sludge_micro.config import ProtocolConfig

    data = config.protocol.model_dump() | {"control_dir": "experiments/stage1"}
    with pytest.raises(ValidationError):
        ProtocolConfig(**data)

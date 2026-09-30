"""Code rules: function length, annotations, import paths, messages module."""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "sludge_micro"
MAX_LINES = 50


def functions():
    for path in sorted(SOURCE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                yield path.name, node


def test_functions_are_at_most_fifty_lines():
    long = [
        f"{name}:{node.name}:{node.end_lineno - node.lineno + 1}"
        for name, node in functions()
        if node.end_lineno - node.lineno + 1 > MAX_LINES
    ]
    assert not long


def test_functions_have_type_annotations():
    missing = []
    for name, node in functions():
        args = [a for a in node.args.args + node.args.kwonlyargs if a.arg not in ("self", "cls")]
        if node.returns is None or any(a.annotation is None for a in args):
            missing.append(f"{name}:{node.name}")
    assert not missing


def test_no_import_path_manipulation():
    for path in SOURCE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "sys.path" not in text, path.name
        assert "PYTHONPATH" not in text, path.name


def test_exceptions_use_message_keys():
    for path in SOURCE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                func = node.exc.func
                if isinstance(func, ast.Name) and func.id == "ProjectError":
                    assert isinstance(node.exc.args[0], ast.Constant), path.name


def test_message_keys_exist():
    from sludge_micro.messages import MESSAGES, STATUS

    keys = set(MESSAGES) | set(STATUS)
    for path in SOURCE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            named = isinstance(node, ast.Call) and getattr(node.func, "id", None) in (
                "ProjectError",
                "text",
            )
            if named and node.args and isinstance(node.args[0], ast.Constant):
                assert node.args[0].value in keys, (path.name, node.args[0].value)

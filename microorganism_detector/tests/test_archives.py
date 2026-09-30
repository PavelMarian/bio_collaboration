"""Safe extraction never escapes the target directory or overwrites other files."""

from __future__ import annotations

import zipfile

import pytest

from sludge_micro.archives import extract_member, safe_member_path, sha256_file
from sludge_micro.messages import ProjectError


@pytest.mark.parametrize("member", ["../evil.txt", "/abs/evil.txt", "a/../../evil.txt"])
def test_unsafe_member_names_are_refused(tmp_path, member):
    with pytest.raises(ProjectError) as info:
        safe_member_path(tmp_path, member, "x.zip")
    assert info.value.key == "archive_unsafe_member"


def test_extract_keeps_identical_and_refuses_conflicting_files(tmp_path, helpers):
    archive = zipfile.ZipFile(helpers.zip_bytes({"dir/a.txt": b"one"}))
    first = extract_member(archive, "dir/a.txt", tmp_path)
    assert first.read_bytes() == b"one"
    assert extract_member(archive, "dir/a.txt", tmp_path) == first
    first.write_bytes(b"changed")
    with pytest.raises(ProjectError) as info:
        extract_member(archive, "dir/a.txt", tmp_path)
    assert info.value.key == "archive_conflict"
    assert first.read_bytes() == b"changed"


def test_sha256_of_file(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

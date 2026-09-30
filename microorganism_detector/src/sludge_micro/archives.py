"""Read-only access to raw archives and safe extraction of copies."""

from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path, PurePosixPath

from sludge_micro.messages import ProjectError

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def open_archive(path: Path) -> zipfile.ZipFile:
    """Open a raw archive for reading.

    Raises:
        ProjectError: If the archive is missing.
    """
    if not path.is_file():
        raise ProjectError("file_missing", path=path, action="положите архив в data/raw.")
    return zipfile.ZipFile(path, "r")


def file_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Return non-directory members sorted by name."""
    return sorted((m for m in archive.infolist() if not m.is_dir()), key=lambda m: m.filename)


def safe_member_path(dest: Path, member: str, archive_name: str) -> Path:
    """Resolve a member path inside ``dest`` or refuse it.

    Raises:
        ProjectError: If the member is absolute or escapes ``dest``.
    """
    pure = PurePosixPath(member)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ProjectError("archive_unsafe_member", member=member, archive=archive_name)
    target = (dest / Path(*pure.parts)).resolve()
    if not target.is_relative_to(dest.resolve()):
        raise ProjectError("archive_unsafe_member", member=member, archive=archive_name)
    return target


def extract_member(archive: zipfile.ZipFile, member: str, dest: Path) -> Path:
    """Extract one member without overwriting a different existing file.

    Args:
        archive: Open source archive.
        member: Member name inside the archive.
        dest: Destination directory for this archive.

    Returns:
        Path of the extracted copy.

    Raises:
        ProjectError: On unsafe member names or conflicting existing files.
    """
    target = safe_member_path(dest, member, str(archive.filename))
    data = archive.read(member)
    if target.exists():
        if target.read_bytes() != data:
            raise ProjectError("archive_conflict", path=target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    temporary.write_bytes(data)
    os.replace(temporary, target)
    return target

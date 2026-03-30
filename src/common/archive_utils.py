"""Helpers for packaging run outputs into zip archives."""

from __future__ import annotations

from pathlib import Path
from shutil import make_archive


def zip_directory_output(source_dir: str | Path) -> Path:
    """Zip a directory into a sibling archive using the same base name."""
    resolved_source = Path(source_dir).resolve()
    if not resolved_source.exists():
        raise FileNotFoundError(f"Cannot archive missing directory: {resolved_source}")
    if not resolved_source.is_dir():
        raise NotADirectoryError(f"Expected directory for archiving: {resolved_source}")

    archive_base = resolved_source.parent / resolved_source.name
    archive_path = Path(make_archive(str(archive_base), "zip", root_dir=str(resolved_source.parent), base_dir=resolved_source.name))
    return archive_path

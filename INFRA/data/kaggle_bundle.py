"""Resolve a LADI input supplied as loose files or one manual-upload ZIP."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile


def resolve_ladi_manifest(expected: str | Path, extraction_dir: str | Path) -> Path:
    """Return the flat manifest, extracting a single safe ZIP if necessary."""
    manifest = Path(expected)
    if manifest.is_file():
        return manifest
    bundle_root = manifest.parent
    archives = sorted(bundle_root.glob("*.zip")) if bundle_root.is_dir() else []
    if len(archives) != 1:
        raise FileNotFoundError(
            f"Expected {manifest} or exactly one ZIP in {bundle_root}; found {len(archives)}")
    destination = Path(extraction_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(archives[0]) as bundle:
        names = bundle.namelist()
        if (len(set(names)) != len(names)
                or any(name != Path(name).name or name in {"", ".", ".."}
                       for name in names)):
            raise ValueError("LADI ZIP must contain unique flat filenames")
        if manifest.name not in names:
            raise ValueError(f"LADI ZIP lacks {manifest.name}")
        bundle.extractall(destination)
    resolved = destination / manifest.name
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved

"""Load the edge-scheduler YAML configs under ``config/edge/``.

One responsibility: locate the repository root and return a config as a plain dict.
No module hard-codes thresholds, paths or label rules; they come from here.
"""
from functools import lru_cache
from pathlib import Path

import yaml

_ROOT_MARKERS = ("AGENTS.md", "pyproject.toml")


def repo_root(start=None):
    """Walk up from ``start`` (default: this file) to the folder that holds the repo markers."""
    here = Path(start or __file__).resolve()
    for parent in (here, *here.parents):
        if all((parent / marker).exists() for marker in _ROOT_MARKERS):
            return parent
    raise FileNotFoundError("repository root not found from " + str(here))


@lru_cache(maxsize=None)
def _read(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(name, root=None):
    """Return ``config/edge/<name>.yaml`` as a dict. ``name`` has no extension."""
    path = Path(root or repo_root()) / "config" / "edge" / (name + ".yaml")
    if not path.is_file():
        raise FileNotFoundError("missing edge config: " + str(path))
    return _read(str(path))

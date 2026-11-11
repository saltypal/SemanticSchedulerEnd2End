"""Build thin Kaggle notebooks for edge steps (nbformat v4 dicts, no nbformat dependency).

Cells only clone/install/call a package entrypoint and write a run manifest; all
logic stays in the package. Never pushes anything (see kaggle_ops.py).
"""
import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/edge/kaggle.yaml"
OUT_DIR = ROOT / "Notebooks/Edge"


_CELL_COUNTER = [0]


def _cell_id(kind, text):
    """Deterministic 12-hex-char id (nbformat 5), unique per cell via a running counter."""
    _CELL_COUNTER[0] += 1
    return hashlib.sha1(f"{_CELL_COUNTER[0]}:{kind}:{text}".encode()).hexdigest()[:12]


def _code(text):
    text = text.strip("\n")
    return {"cell_type": "code", "execution_count": None, "id": _cell_id("code", text), "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def _md(text):
    return {"cell_type": "markdown", "id": _cell_id("md", text), "metadata": {},
            "source": text.splitlines(keepends=True)}


def title_for_slug(slug):
    """edge-a-download-images -> 'Edge A Download Images' (single-letter words upper-case)."""
    return " ".join(w.upper() if len(w) == 1 else w.capitalize() for w in slug.split("-") if w)


def _env_cell():
    return _code('''
import os, time, warnings
os.environ.update(PYTHONWARNINGS='ignore', PYDEVD_DISABLE_FILE_VALIDATION='1', TOKENIZERS_PARALLELISM='false',
                  HF_HUB_DISABLE_PROGRESS_BARS='1', TQDM_DISABLE='1')
warnings.filterwarnings('ignore')
T0 = time.time()
''')


def _clone_cell(repo_url, branch):
    return _code(f'''
import os, subprocess
REPO, URL, BRANCH = "/kaggle/working/repo", "{repo_url}", "{branch}"
def _git(*a, cwd=None):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
if os.path.isdir(REPO + "/.git"):
    _git("fetch", "-q", "--depth", "1", "origin", BRANCH, cwd=REPO)
    _git("reset", "-q", "--hard", "FETCH_HEAD", cwd=REPO)
else:
    _git("clone", "--depth", "1", "-q", "--no-progress", "--branch", BRANCH, URL, REPO)
GIT_SHA = _git("rev-parse", "HEAD", cwd=REPO)
print("repo @", GIT_SHA[:8], "(" + _git("log", "-1", "--format=%s", cwd=REPO) + ")")
''')


def _install_cell(requirements):
    reqs = list(map(str, requirements))
    return _code(f'''
import subprocess, time
_t = time.time()
LOG = "/kaggle/working/pip_install.log"
with open(LOG, "w") as _f:
    _rc = subprocess.run(["pip", "install", "-q", "--no-input", "--disable-pip-version-check",
                          "--ignore-requires-python", "-e", ".", *{reqs!r}],
                         cwd=REPO, stdout=_f, stderr=subprocess.STDOUT).returncode
if _rc:
    print("".join(open(LOG).readlines()[-25:]))
    raise RuntimeError("pip install failed (rc=%s)" % _rc)
print("installed in %.0fs" % (time.time() - _t))
''')


def _run_cell(entrypoint_module, args):
    argv = ["python", "-u", "-m", entrypoint_module, *map(str, args)]
    return _code(f'''
import os, subprocess
ARGV = {argv!r}
print("$ python -m {entrypoint_module} " + " ".join(ARGV[4:]))
_p = subprocess.Popen(ARGV, cwd=REPO, env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for _line in _p.stdout:
    print(_line, end="", flush=True)
if _p.wait():
    raise RuntimeError("{entrypoint_module} exited with %s" % _p.returncode)
''')


def _manifest_cell(step_id, entrypoint_module, args):
    return _code(f'''
import datetime, json, os, platform, subprocess, time
STEP, ARGS = "{step_id}", {list(map(str, args))!r}
def _gpu():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    except Exception:
        return None
try:
    import torch; torch_version = torch.__version__
except Exception:
    torch_version = None
out = "/kaggle/working/runs/" + STEP
os.makedirs(out, exist_ok=True)
manifest = {{"step": STEP, "entrypoint": "{entrypoint_module}", "args": ARGS, "git_sha": GIT_SHA,
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "python": platform.python_version(),
            "torch": torch_version, "gpu": _gpu(), "wall_time_s": time.time() - T0}}
json.dump(manifest, open(out + "/run_manifest.json", "w"), indent=2)
print(manifest)
''')


def build_notebook(step_id, entrypoint_module, args, title, repo_url, branch="main",
                   requirements=(), needs_gpu=True):
    """Return an nbformat v4 notebook dict: clone, install, run, manifest cells."""
    _CELL_COUNTER[0] = 0
    cells = [_md(f"# {title}\n\nStep `{step_id}`. Thin wrapper; logic lives in `{entrypoint_module}`.\n"),
             _env_cell(), _clone_cell(repo_url, branch), _install_cell(requirements),
             _run_cell(entrypoint_module, args), _manifest_cell(step_id, entrypoint_module, args)]
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu" if needs_gpu else "none"}}
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}


def load_kaggle_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text())


def kernel_slug(cfg, step_id):
    """Step ids that already start with the prefix (e.g. edge-pilot) are not prefixed twice."""
    prefix = cfg.get("slug_prefix", "edge-")
    return step_id if step_id.startswith(prefix) else f"{prefix}{step_id}"


def build_metadata(cfg, account, step_id, title, needs_gpu=True, dataset_sources=(), kernel_sources=()):
    """kernel-metadata.json dict; owner comes from the account's username in config."""
    owner = cfg["accounts"][account].get("username", "")
    return {"id": f"{owner}/{kernel_slug(cfg, step_id)}", "title": title,
            "code_file": f"{step_id}.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": bool(needs_gpu), "enable_internet": True,
            "dataset_sources": list(dataset_sources), "kernel_sources": list(kernel_sources),
            "competition_sources": []}


def write_outputs(nb, meta, step_id, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_path = out_dir / f"{step_id}.ipynb"
    meta_path = out_dir / f"{step_id}.kernel-metadata.json"
    nb_path.write_text(json.dumps(nb, indent=1))
    meta_path.write_text(json.dumps(meta, indent=2))
    return nb_path, meta_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", required=True)
    p.add_argument("--entrypoint", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--account", default=None, choices=("win", "wsl"), help="default: default_account")
    p.add_argument("--branch", default="main")
    p.add_argument("--requirement", action="append", default=[])
    p.add_argument("--dataset-source", action="append", default=[])
    p.add_argument("--kernel-source", action="append", default=[])
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("args", nargs=argparse.REMAINDER, help="after --: entrypoint args")
    a = p.parse_args(argv)
    cfg = load_kaggle_config()
    ep_args = [x for x in a.args if x != "--"]
    nb = build_notebook(a.step, a.entrypoint, ep_args, a.title, cfg["repo_url"], a.branch,
                        tuple(a.requirement), not a.no_gpu)
    account = a.account or cfg.get("default_account", "wsl")
    meta = build_metadata(cfg, account, a.step, a.title, not a.no_gpu, a.dataset_source, a.kernel_source)
    for path in write_outputs(nb, meta, a.step, a.out_dir):
        print(path)


if __name__ == "__main__":
    main()

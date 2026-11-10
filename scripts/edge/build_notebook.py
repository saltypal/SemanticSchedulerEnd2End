"""Build thin Kaggle notebooks for edge steps (nbformat v4 dicts, no nbformat dependency).

Cells only clone/install/call a package entrypoint and write a run manifest; all
logic stays in the package. Never pushes anything (see kaggle_ops.py).
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/edge/kaggle.yaml"
OUT_DIR = ROOT / "Notebooks/Edge"


def _code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def _md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _clone_cell(repo_url, branch):
    return _code(f'''
import os, subprocess
REPO, URL, BRANCH = "/kaggle/working/repo", "{repo_url}", "{branch}"
if os.path.isdir(REPO + "/.git"):
    subprocess.run(["git", "-C", REPO, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO, "reset", "--hard", "origin/" + BRANCH], check=True)
else:
    subprocess.run(f"git clone --depth 1 --branch {{BRANCH}} {{URL}} {{REPO}}", shell=True, check=True)
GIT_SHA = subprocess.check_output(f"git -C {{REPO}} rev-parse HEAD", shell=True, text=True).strip()
print("commit", GIT_SHA)
''')


def _install_cell(requirements):
    extra = "".join(f" {r}" for r in requirements)
    return _code(f'''
%cd /kaggle/working/repo
!pip install -q -e .{extra}
''')


def _run_cell(entrypoint_module, args):
    cmd = " ".join(["python", "-u", "-m", entrypoint_module, *map(str, args)])
    return _code(f"!{cmd}")


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
    cells = [_md(f"# {title}\n\nStep `{step_id}`. Thin wrapper; logic lives in `{entrypoint_module}`.\n"),
             _clone_cell(repo_url, branch), _install_cell(requirements),
             _code("import time; T0 = time.time()"),
             _run_cell(entrypoint_module, args), _manifest_cell(step_id, entrypoint_module, args)]
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu" if needs_gpu else "none"}}
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}


def load_kaggle_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text())


def kernel_slug(cfg, step_id):
    return f"{cfg.get('slug_prefix', 'edge-')}{step_id}"


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

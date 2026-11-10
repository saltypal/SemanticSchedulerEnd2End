"""Kaggle CLI wrapper for edge steps: kernel push/status/wait/pull/list and dataset
create/version/status/list, with an injectable runner.

Accounts come from config/edge/kaggle.yaml (config_dir -> KAGGLE_CONFIG_DIR, bin ->
executable, runner wsl -> `wsl <bin> ...` with Windows -p/--path values translated to
/mnt/<drive>/...). --account defaults to default_account. --dry-run prints commands only.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.edge.build_notebook import OUT_DIR, ROOT, kernel_slug, load_kaggle_config

RUN_LOG = ROOT / "parameters/edge_kaggle_runs.md"
LOG_HEADER = "# Edge Kaggle runs\n\n| step | account | slug | version | git SHA | utc |\n|---|---|---|---|---|---|\n"
TERMINAL = ("complete", "error", "cancel")
ACCOUNTS = ("win", "wsl")
_WIN_PATH = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_PATH_FLAGS = ("-p", "--path")


def to_wsl_path(p):
    """Translate a Windows path (D:\\x or D:/x) to /mnt/d/x; other strings pass through."""
    m = _WIN_PATH.match(str(p))
    if not m:
        return str(p)
    return f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")


def real_runner(cmd, env=None):
    """Run cmd (no shell); return (returncode, combined stdout)."""
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    return r.returncode, (r.stdout or "") + (r.stderr or "")


class Ops:
    def __init__(self, cfg, account, runner=real_runner, dry_run=False, out=print, log_path=RUN_LOG,
                 stage_root=OUT_DIR):
        self.cfg, self.account, self.runner = cfg, account, runner
        self.dry_run, self.out, self.log_path = dry_run, out, Path(log_path)
        self.stage_root = stage_root
        self.acc = cfg["accounts"][account]

    def slug(self, step):
        return f"{self.acc.get('username', '')}/{kernel_slug(self.cfg, step)}"

    def _env(self):
        d = self.acc.get("config_dir")
        return {"KAGGLE_CONFIG_DIR": str(d)} if d else {}

    def _cmd(self, args):
        binary = self.acc.get("bin") or "kaggle"
        args = [str(a) for a in args]
        if self.acc.get("runner") != "wsl":
            return [binary, *args]
        for i in range(1, len(args)):
            if args[i - 1] in _PATH_FLAGS:
                args[i] = to_wsl_path(args[i])
        return ["wsl", binary, *args]

    def run(self, args):
        cmd = self._cmd(args)
        self.out("$ " + " ".join(cmd) + (f"  [env {self._env()}]" if self._env() else ""))
        if self.dry_run:
            return 0, ""
        if not self.acc.get("username"):
            raise ValueError("kaggle account %r has no username in config/edge/kaggle.yaml" % self.account)
        rc, text = self.runner(cmd, self._env())
        self.out(text)
        return rc, text

    def git_sha(self):
        rc, text = self.runner(["git", "-C", str(ROOT), "rev-parse", "HEAD"], {})
        return text.strip() if rc == 0 else "unknown"

    def append_log(self, step, version="", slug=None):
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text(LOG_HEADER)
        utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        row = f"| {step} | {self.account} | {slug or self.slug(step)} | {version} | {self.git_sha()} | {utc} |\n"
        with self.log_path.open("a") as f:
            f.write(row)

    def stage(self, step, nb_dir=OUT_DIR):
        """Copy notebook + account-specific metadata into a staging folder."""
        stage = Path(nb_dir) / "_stage" / self.account / step
        if self.dry_run:
            return stage
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(nb_dir) / f"{step}.ipynb", stage / f"{step}.ipynb")
        meta = json.loads((Path(nb_dir) / f"{step}.kernel-metadata.json").read_text())
        meta["id"] = self.slug(step)
        (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        return stage

    def push(self, step, nb_dir=OUT_DIR):
        stage = self.stage(step, nb_dir)
        rc, text = self.run(["kernels", "push", "-p", str(stage)])
        if rc == 0 and not self.dry_run:
            m = re.search(r"[Vv]ersion\s+(\d+)", text)
            self.append_log(step, m.group(1) if m else "")
        return rc

    def status_text(self, step):
        return self.run(["kernels", "status", self.slug(step)])[1].lower()

    def status(self, step):
        self.status_text(step)
        return 0

    def wait(self, step, timeout=3600, poll=30, sleep=time.sleep):
        deadline = time.monotonic() + timeout
        while True:
            text = self.status_text(step)
            if self.dry_run or any(t in text for t in TERMINAL):
                return 1 if "error" in text else 0
            if time.monotonic() >= deadline:
                return 2
            sleep(poll)

    def pull(self, step, out):
        rc, _ = self.run(["kernels", "output", self.slug(step), "-p", str(out)])
        if rc == 0 and not self.dry_run:
            self.append_log(step, "pull")
        return rc

    def list(self):
        return self.run(["kernels", "list", "--mine", "--search", self.cfg.get("slug_prefix", "edge-")])[0]

    # datasets
    def dataset_slug(self, slug):
        return slug if "/" in slug else f"{self.acc.get('username', '')}/{slug}"

    def stage_dataset(self, src, title=None, slug=None):
        """Copy src into a staging folder; write dataset-metadata.json there only if missing."""
        src = Path(src)
        name = slug or re.sub(r"[^a-z0-9-]+", "-", src.name.lower()).strip("-")
        stage = Path(self.stage_root) / "_stage" / self.account / "datasets" / name
        if self.dry_run:
            return stage, name
        shutil.copytree(src, stage, dirs_exist_ok=True)
        meta = stage / "dataset-metadata.json"
        if not meta.exists():
            meta.write_text(json.dumps({"id": self.dataset_slug(name), "title": title or name,
                                        "licenses": [{"name": "other"}]}, indent=2))
        return stage, name

    def _dataset_push(self, verb, src, title, slug, extra, dir_mode):
        stage, name = self.stage_dataset(src, title, slug)
        args = ["datasets", verb, "-p", str(stage), *extra]
        if dir_mode:
            args += ["--dir-mode", dir_mode]
        rc, _ = self.run(args)
        if rc == 0 and not self.dry_run:
            self.append_log(f"dataset-{verb}", "", self.dataset_slug(name))
        return rc

    def dataset_create(self, src, title=None, public=False, dir_mode=None, slug=None):
        return self._dataset_push("create", src, title, slug, ["--public"] if public else [], dir_mode)

    def dataset_version(self, src, message, dir_mode=None, slug=None):
        return self._dataset_push("version", src, None, slug, ["-m", message], dir_mode)

    def dataset_status(self, slug):
        return self.run(["datasets", "status", self.dataset_slug(slug)])[0]

    def dataset_list(self):
        return self.run(["datasets", "list", "--mine"])[0]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name):
        s = sub.add_parser(name)
        s.add_argument("--account", default=None, choices=ACCOUNTS, help="default: default_account in config")
        return s

    for name in ("push", "status", "wait", "pull", "list"):
        s = add(name)
        if name != "list":
            s.add_argument("--step", required=True)
        if name == "wait":
            s.add_argument("--timeout", type=int, default=3600)
            s.add_argument("--poll", type=int, default=30)
        if name == "pull":
            s.add_argument("--out", required=True)
    for name in ("dataset-create", "dataset-version"):
        s = add(name)
        s.add_argument("--dir", required=True)
        s.add_argument("--dir-mode", choices=("tar",), default=None)
        s.add_argument("--slug", default=None)
        if name == "dataset-create":
            s.add_argument("--title", default=None)
            s.add_argument("--public", action="store_true")
        else:
            s.add_argument("--message", required=True)
    add("dataset-status").add_argument("--slug", required=True)
    add("dataset-list").add_argument("--mine", action="store_true", default=True)
    return p


def dispatch(a, ops):
    if a.cmd == "push":
        return ops.push(a.step)
    if a.cmd == "status":
        return ops.status(a.step)
    if a.cmd == "wait":
        return ops.wait(a.step, a.timeout, a.poll)
    if a.cmd == "pull":
        return ops.pull(a.step, a.out)
    if a.cmd == "dataset-create":
        return ops.dataset_create(a.dir, a.title, a.public, a.dir_mode, a.slug)
    if a.cmd == "dataset-version":
        return ops.dataset_version(a.dir, a.message, a.dir_mode, a.slug)
    if a.cmd == "dataset-status":
        return ops.dataset_status(a.slug)
    if a.cmd == "dataset-list":
        return ops.dataset_list()
    return ops.list()


def resolve_account(cfg, account):
    return account or cfg.get("default_account") or "wsl"


def main(argv=None, runner=real_runner, cfg=None, log_path=RUN_LOG, out=print, stage_root=OUT_DIR):
    a = build_parser().parse_args(argv)
    cfg = cfg or load_kaggle_config()
    ops = Ops(cfg, resolve_account(cfg, a.account), runner, a.dry_run, out, log_path, stage_root)
    return dispatch(a, ops)


if __name__ == "__main__":
    sys.exit(main())

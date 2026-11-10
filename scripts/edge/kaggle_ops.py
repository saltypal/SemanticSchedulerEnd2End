"""Kaggle CLI wrapper for edge steps: push/status/wait/pull/list with injectable runner.

Accounts come from config/edge/kaggle.yaml (config_dir -> KAGGLE_CONFIG_DIR,
runner wsl -> commands prefixed with `wsl`). --dry-run prints commands only.
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


def real_runner(cmd, env=None):
    """Run cmd; return (returncode, combined stdout)."""
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    return r.returncode, (r.stdout or "") + (r.stderr or "")


class Ops:
    def __init__(self, cfg, account, runner=real_runner, dry_run=False, out=print, log_path=RUN_LOG):
        self.cfg, self.account, self.runner = cfg, account, runner
        self.dry_run, self.out, self.log_path = dry_run, out, Path(log_path)
        self.acc = cfg["accounts"][account]

    def slug(self, step):
        return f"{self.acc.get('username', '')}/{kernel_slug(self.cfg, step)}"

    def _env(self):
        d = self.acc.get("config_dir")
        return {"KAGGLE_CONFIG_DIR": str(d)} if d else {}

    def _cmd(self, args):
        prefix = ["wsl"] if self.acc.get("runner") == "wsl" else []
        return [*prefix, "kaggle", *args]

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

    def append_log(self, step, version=""):
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text(LOG_HEADER)
        utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        row = f"| {step} | {self.account} | {self.slug(step)} | {version} | {self.git_sha()} | {utc} |\n"
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


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("push", "status", "wait", "pull", "list"):
        s = sub.add_parser(name)
        s.add_argument("--account", required=True, choices=("win", "wsl"))
        if name != "list":
            s.add_argument("--step", required=True)
        if name == "wait":
            s.add_argument("--timeout", type=int, default=3600)
            s.add_argument("--poll", type=int, default=30)
        if name == "pull":
            s.add_argument("--out", required=True)
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
    return ops.list()


def main(argv=None, runner=real_runner, cfg=None, log_path=RUN_LOG, out=print):
    a = build_parser().parse_args(argv)
    ops = Ops(cfg or load_kaggle_config(), a.account, runner, a.dry_run, out, log_path)
    return dispatch(a, ops)


if __name__ == "__main__":
    sys.exit(main())

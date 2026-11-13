"""Emit Notebooks/Edge/edge-scheduler-sweeps.ipynb (or ``step``): one named stage per notebook version.

Stages (``Evaluation.edge.sweep_stages``): tune, compare, ablations, test (test needs ``--final-test`` and a frozen
``--policies`` set). ``--shard i/n`` splits a big sweep across notebooks/accounts. Policies may be ``learned:<algo>:<ckpt>``;
a bare checkpoint file name is found under the attached kernel outputs (``--kernel-source``). ``--sts-fusion``
passes ``sts_fusion.json`` (found by name next to the sim table) to the sweep. Never pushes anything.
"""
import argparse

from scripts.edge import _nb_common as nc
from scripts.edge import build_notebook as bn

STEP = "edge-scheduler-sweeps"
TITLE = "Edge Scheduler Sweeps"
ENTRY = "Evaluation.edge.sweep_stages"
STAGES = ("tune", "compare", "ablations", "test")
DATASET_SOURCES = (nc.SIMTABLE_DS,)

RUN = '''
import os, subprocess
OUT = "/kaggle/working/runs/" + {step!r}
os.makedirs(OUT, exist_ok=True)
EXTRA, STS = {extra!r}, {sts!r}
TUNED = {tuned!r}


def _resolve(a):
    """learned:<algo>:<file name> -> learned:<algo>:<path under /kaggle/input>."""
    if isinstance(a, str) and a.startswith("learned:"):
        algo, _, ck = a[len("learned:"):].partition(":")
        if ck and "/" not in ck and chr(92) not in ck:
            ck = find_input(ck)
        return "learned:%s:%s" % (algo, ck)
    return a


EXTRA = [_resolve(x) for x in EXTRA] + (["--sts-fusion", find_input("sts_fusion.json")] if STS else [])
if TUNED:   # tuned SV-DPP kwargs travel inside the notebook (no extra dataset)
    import json
    json.dump(TUNED, open(OUT + "/tuned_in.json", "w"))
    EXTRA += ["--tuned", OUT + "/tuned_in.json"]
ARGV = ["python", "-u", "-m", {entry!r}, "--stage", {stage!r}, "--table", TABLE, "--profile", PROFILE,
        "--out-dir", OUT] + EXTRA
print("$", " ".join(ARGV[3:]), flush=True)
p = subprocess.Popen(ARGV, cwd=REPO, env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="", flush=True)
if p.wait():
    raise RuntimeError("sweep stage {stage} failed (rc=%d)" % p.returncode)
print(sorted(os.listdir(OUT)))
'''


def stage_args(stage, shard="", policies=(), final_test=False):
    extra = []
    if shard:
        extra += ["--shard", shard]
    if policies:
        extra += ["--policies", *policies]
    if stage == "test":
        if not final_test:
            raise ValueError("stage 'test' needs --final-test")
        if not policies:
            raise ValueError("stage 'test' needs the frozen --policies set")
        extra.append("--final-test")
    elif final_test:
        raise ValueError("--final-test only applies to stage 'test'")
    return extra


def build(stage="compare", account=None, branch="main", shard="", policies=(), final_test=False, step=None,
          sts_fusion=False, kernel_sources=(), extra_args=(), tuned=None):
    if stage not in STAGES:
        raise ValueError("unknown stage %r" % stage)
    extra = stage_args(stage, shard, policies, final_test) + [str(x) for x in extra_args]
    step = step or STEP
    title = TITLE if step == STEP else bn.title_for_slug(step)
    cfg = bn.load_kaggle_config()
    nb = bn.build_notebook(step, ENTRY, ["--stage", stage, *extra, *(["--sts-fusion"] if sts_fusion else [])], title, cfg["repo_url"], branch,
                           ("scikit-learn", "PyYAML"), False)
    nb["cells"][0] = bn._md(f"# {title}\n\nStep `{step}`, stage `{stage}`{', shard ' + shard if shard else ''}. "
                            f"Logic lives in `{ENTRY}`.\n\n{nc.PRIVATE_NOTE}")
    nb["cells"][4:5] = [nc.discover_cell(), bn._code(RUN.format(step=step, entry=ENTRY, stage=stage, extra=extra, sts=bool(sts_fusion), tuned=tuned))]
    meta = bn.build_metadata(cfg, account or cfg.get("default_account", "wsl"), step, title, False, DATASET_SOURCES, kernel_sources)
    return nc.finish(nb, meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--shard", default="", help="i/n, e.g. 0/2")
    ap.add_argument("--policies", nargs="+", default=[])
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--sts-fusion", action="store_true", help="pass sts_fusion.json (found by name) to the sweep")
    ap.add_argument("--kernel-source", action="append", default=[], help="kernel whose outputs hold learned checkpoints")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--account", default=None, choices=("win", "wsl"))
    ap.add_argument("--out-dir", default=str(bn.OUT_DIR))
    a = ap.parse_args(argv)
    nb, meta = build(a.stage, a.account, a.branch, a.shard, a.policies, a.final_test, None, a.sts_fusion,
                     a.kernel_source)
    nc.write(nb, meta, STEP, a.out_dir)


if __name__ == "__main__":
    main()

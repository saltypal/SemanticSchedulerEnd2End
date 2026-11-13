"""Emit Notebooks/Edge/edge-scheduler-train.ipynb (+ kernel metadata): one learner per notebook version.

``--algo`` picks the learner (ppo, scope_ppo, bc, dagger, dqn, scope_dqn, ltr, scope_ltr, semcover, bcedge;
``--ranker`` for ltr). ``step`` gives each learner its own slug (launch_phase2: edge-train-<algo>).
The run cell calls ``python -m Scheduler.training.edge.train_cli`` once per seed on the train split of the sim
table with the latency profile; checkpoints and learning curves go to /kaggle/working/runs/<step>/. Never pushes.
"""
import argparse

from scripts.edge import _nb_common as nc
from scripts.edge import build_notebook as bn

STEP = "edge-scheduler-train"
TITLE = "Edge Scheduler Train"      # slugifies to STEP (MISTAKES 26)
ENTRY = "Scheduler.training.edge.train_cli"
ALGOS = ("ppo", "scope_ppo", "bc", "dagger", "dqn", "scope_dqn", "ltr", "scope_ltr", "semcover", "bcedge")
DATASET_SOURCES = (nc.SIMTABLE_DS,)

RUN = '''
import os, subprocess
ALGO, SEEDS, RANKER, EXTRA = {algo!r}, {seeds!r}, {ranker!r}, {extra!r}
OUT = "/kaggle/working/runs/" + {step!r}
os.makedirs(OUT, exist_ok=True)
for seed in SEEDS:
    ckpt = "%s/%s_seed%d.pt" % (OUT, ALGO, seed)
    argv = ["python", "-u", "-m", {entry!r}, "--algo", ALGO, "--table", TABLE, "--profile", PROFILE,
            "--out", ckpt, "--seed", str(seed)] + (["--ranker", RANKER] if ALGO == "ltr" else []) + EXTRA
    print("$", " ".join(argv[3:]), flush=True)
    p = subprocess.Popen(argv, cwd=REPO, env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise RuntimeError("train_cli failed for seed %d (rc=%d)" % (seed, p.returncode))
print(sorted(os.listdir(OUT)))
'''


def build(algo="ppo", account=None, branch="main", seeds=5, ranker="lambdamart", extra=(), step=None):
    if algo not in ALGOS:
        raise ValueError("unknown algo %r; choose from %s" % (algo, ALGOS))
    step = step or STEP
    title = TITLE if step == STEP else bn.title_for_slug(step)
    cfg = bn.load_kaggle_config()
    seed_list = list(range(int(seeds)))
    nb = bn.build_notebook(step, ENTRY, ["--algo", algo, "--seeds", seed_list], title, cfg["repo_url"], branch,
                           ("scikit-learn", "lightgbm", "PyYAML"), False)
    nb["cells"][0] = bn._md(f"# {title}\n\nStep `{step}`, learner `{algo}`, seeds {seed_list}. "
                            f"Logic lives in `{ENTRY}`.\n\n{nc.PRIVATE_NOTE}")
    run = bn._code(RUN.format(algo=algo, seeds=seed_list, ranker=ranker, extra=list(extra), step=step, entry=ENTRY))
    nb["cells"][4:5] = [nc.discover_cell(), run]                 # replace the generic run cell
    meta = bn.build_metadata(cfg, account or cfg.get("default_account", "wsl"), step, title, False, DATASET_SOURCES)
    return nc.finish(nb, meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", required=True, choices=ALGOS)
    ap.add_argument("--ranker", default="lambdamart", choices=("lambdamart", "ranknet"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--account", default=None, choices=("win", "wsl"))
    ap.add_argument("--out-dir", default=str(bn.OUT_DIR))
    ap.add_argument("extra", nargs="*", help="extra train_cli args after --")
    a = ap.parse_args(argv)
    nb, meta = build(a.algo, a.account, a.branch, a.seeds, a.ranker, [x for x in a.extra if x != "--"])
    nc.write(nb, meta, STEP, a.out_dir)


if __name__ == "__main__":
    main()

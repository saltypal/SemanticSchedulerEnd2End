"""Named sweep stages for the phase-2 notebook: ``python -m Evaluation.edge.sweep_stages --stage S ...``.

Stages: ``tune`` (val: SV-DPP V x hard_purge_prob grid, optional learner training grid through
train_cli), ``compare`` (val: main_table + extended, c in 1,2,4,6,8, 10 realisations),
``ablations`` (header proofs, delegated to Evaluation.edge.header_proofs when it exists) and
``test`` (frozen policy set, only with ``--final-test``). ``--shard i/n`` splits the work over
accounts/notebooks (policies, or grid points for ``tune``) and writes ``<stage>.shard<i>of<n>.json``.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from Evaluation.edge import run_sweep as rs
from Evaluation.edge.contracts import ValueSpec
from Evaluation.edge.edge_config import load_config

STAGES = ("tune", "compare", "ablations", "test")
WORKERS = [1, 2, 4, 6, 8]
RHOS = [0.8, 1.0, 1.2]
V_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
HPP_GRID = [0.9, 0.95, 0.99]


def parse_shard(text):
    """'i/n' (0-based i) -> (i, n); '' -> (0, 1)."""
    if not text:
        return 0, 1
    i, n = (int(x) for x in text.split("/"))
    if n < 1 or not 0 <= i < n:
        raise ValueError("bad shard %r: need 0 <= i < n" % text)
    return i, n


def take_shard(items, shard):
    i, n = shard
    return list(items)[i::n]


def out_path(out_dir, stage, shard):
    name = stage if shard[1] == 1 else "%s.shard%dof%d" % (stage, shard[0], shard[1])
    return Path(out_dir) / (name + ".json")


def compare_policies():
    cfg = load_config("policies")
    return list(dict.fromkeys(cfg["main_table"] + cfg["extended"]))


def stage_tune(rows, latency, a, shard, tables):
    spec = ValueSpec.from_config(load_config("sim"))
    rows_, spec, mb = rs._setup(rows, latency, "val", spec, a.max_batch)
    grid = take_shard([(v, h) for v in V_GRID for h in HPP_GRID], shard)
    points = [rs.run_cell(rows_, latency, spec, "sv_dpp", dict(V=v, hard_purge_prob=h), 2, 1.0,
                          a.realisations, a.seed, mb, None, 0.0, a.n_boot, tables) for v, h in grid]
    ok = [p for p in points if p["ctr_ungated"] == p["ctr_ungated"]]
    best = max(ok, key=lambda p: p["ctr_ungated"]) if ok else None
    out = dict(stage="tune", shard=list(shard), sv_dpp_grid=points, best_kwargs=best and best["kwargs"], learners=[])
    if a.learner_grid:
        from Scheduler.training.edge import train_cli
        for k, entry in enumerate(take_shard(json.loads(Path(a.learner_grid).read_text()), shard)):
            ckpt = str(Path(a.out_dir) / ("tune_learner_%d_%s.pt" % (k, entry["algo"])))
            argv = ["--table", a.table, "--profile", a.profile, "--out", ckpt] + [str(x) for x in entry["args"]]
            hist = train_cli.main(["--algo", entry["algo"]] + argv)
            out["learners"].append(dict(entry=entry, checkpoint=ckpt, history_tail=hist[-3:] if isinstance(hist, list) else None))
    return out


def stage_compare(rows, latency, a, shard, tables):
    policies = take_shard(rs.resolve_policies(a.policies or compare_policies()), shard)
    kw = {}
    if a.tuned:
        best = json.loads(Path(a.tuned).read_text()).get("best_kwargs")
        if best:
            kw["sv_dpp"] = best
    return rs.run_sweep(rows, latency, "val", policies, a.workers, a.rhos, a.realisations, a.seed,
                        n_boot=a.n_boot, policy_kwargs=kw, tables=tables)


def stage_test(rows, latency, a, shard, tables):
    if not a.final_test:
        raise SystemExit("stage 'test' touches the test split: pass --final-test (policy set must be frozen)")
    if not a.policies:
        raise SystemExit("stage 'test' needs the frozen policy set via --policies")
    policies = take_shard(rs.resolve_policies(a.policies), shard)
    kw = {}
    if a.tuned:
        best = json.loads(Path(a.tuned).read_text()).get("best_kwargs")
        if best:
            kw["sv_dpp"] = best
    return rs.run_sweep(rows, latency, "test", policies, a.workers, a.rhos, a.realisations, a.seed,
                        n_boot=a.n_boot, policy_kwargs=kw, tables=tables)


def stage_ablations(a, shard):
    """Header proofs live in Evaluation/edge/header_proofs.py (W-SIM). Fail loudly if absent."""
    if importlib.util.find_spec("Evaluation.edge.header_proofs") is None:
        raise SystemExit("ABLATIONS NOT AVAILABLE: Evaluation/edge/header_proofs.py is not on this branch yet "
                         "(owned by W-SIM). Re-run after it lands on main.")
    dest = out_path(a.out_dir, "ablations", shard).with_suffix(".proofs.json")
    # ASSUMED CLI: --table --profile --out (adjust here if header_proofs uses other flags).
    cmd = [sys.executable, "-u", "-m", "Evaluation.edge.header_proofs", "--table", a.table,
           "--profile", a.profile, "--out", str(dest)]
    if subprocess.run(cmd).returncode:
        raise SystemExit("header_proofs failed: " + " ".join(cmd))
    return dict(stage="ablations", shard=list(shard), proofs_file=str(dest))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--table", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", default="", help="i/n (0-based)")
    ap.add_argument("--policies", nargs="+", default=None)
    ap.add_argument("--workers", nargs="+", type=int, default=WORKERS)
    ap.add_argument("--rhos", nargs="+", type=float, default=RHOS)
    ap.add_argument("--realisations", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--max-batch", type=int, default=None)
    ap.add_argument("--tuned", default=None, help="tune-stage JSON whose best_kwargs go to sv_dpp")
    ap.add_argument("--learner-grid", default=None, help="JSON list of {algo, args:[...]} for the tune stage")
    ap.add_argument("--final-test", action="store_true")
    a = ap.parse_args(argv)
    shard = parse_shard(a.shard)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    if a.stage == "ablations":
        res = stage_ablations(a, shard)
    else:
        from Evaluation.edge import tables
        rows, latency = tables.load_rows(a.table), tables.latency_from_profile_json(a.profile)
        fn = dict(tune=stage_tune, compare=stage_compare, test=stage_test)[a.stage]
        res = fn(rows, latency, a, shard, tables)
        res["git_sha"] = rs.git_sha()
    rs.save_results(res, out_path(a.out_dir, a.stage, shard))
    return res


if __name__ == "__main__":
    main()

"""Overnight orchestrator: dry-run of every stage (no Kaggle, no network, no subprocess), done-markers / skip,
failure isolation, STATUS.md lines, and the pure helpers (curves, tuning pick, header verdict, merge)."""
import json

from scripts.edge import run_overnight as ro

ALL = [s[0] for s in ro.STAGES]


def status_lines(out):
    return (out / "STATUS.md").read_text(encoding="utf-8").splitlines()


def test_dry_run_all_stages_order_markers_and_skip(tmp_path):
    ctx = ro.Ctx(tmp_path, dry=True, parallel=2, sleep=lambda s: None, scratch=tmp_path / "scr")
    res = ro.run_stages(ctx)
    assert res == {s: "ok" for s in ALL}
    assert ctx.out == tmp_path / "dry_run"
    for s in ALL:
        assert (ctx.out / (s + ".done.json")).exists()
    ok = [ln for ln in status_lines(ctx.out) if "| OK |" in ln]
    assert [ln.split("|")[1].split()[0] for ln in ok] == ALL        # stage ordering, one line each
    assert all(ln.startswith("- 20") and ln.count("|") >= 3 for ln in ok)
    cmds = (ctx.out / "dry_commands.log").read_text(encoding="utf-8")
    assert "decision_trace" in cmds and "header_proofs" in cmds and "sweep_stages" in cmds
    assert "--final-test" in cmds and "--sts-fusion" in cmds and "--budget" in cmds
    res2 = ro.run_stages(ctx)                                        # second run: everything skipped
    assert res2 == {s: "skip" for s in ALL}
    assert sum("| SKIP |" in ln for ln in status_lines(ctx.out)) == len(ALL)


def test_only_and_failure_isolation(tmp_path):
    ctx = ro.Ctx(tmp_path, dry=True, sleep=lambda s: None, scratch=tmp_path / "scr")

    def boom(c):
        raise RuntimeError("kaboom")

    def fine(c):
        return "fine"
    stages = (("S1", "a", fine), ("S2", "b", boom), ("S3", "c", fine))
    res = ro.run_stages(ctx, stages=stages)
    assert res == {"S1": "ok", "S2": "fail", "S3": "ok"}
    assert not (ctx.out / "S2.done.json").exists() and (ctx.out / "S3.done.json").exists()
    assert any("| FAIL |" in ln and "kaboom" in ln for ln in status_lines(ctx.out))
    ro.run_stages(ctx, only={"S2"}, stages=stages)                   # failed stage is retried, others not touched
    assert sum("S3" in ln and "| OK |" in ln for ln in status_lines(ctx.out)) == 1


def test_main_cli_only_dry(tmp_path, monkeypatch):
    monkeypatch.setattr(ro, "TMP", tmp_path / "tmp")
    res = ro.main(["--only", "S3", "S9", "--dry-run", "--out", str(tmp_path), "--parallel", "2"])
    assert res == {"S3": "ok", "S9": "ok"}
    assert (tmp_path / "dry_run" / "FINAL_REPORT.md").exists()


def hist(vals):
    return [dict(iter=i, mean_reward=v, tier_success=[0.5, 0.6, 0.7, 0.8]) for i, v in enumerate(vals)]


def test_convergence_flag_and_curves(tmp_path):
    flat = [1 + 0.5 * min(i, 20) / 20 for i in range(100)]            # saturates at i=20
    rising = [i * 0.1 for i in range(100)]
    assert not ro.not_converged(flat) and ro.not_converged(rising)
    p2 = tmp_path / "p2"
    for slug, vals in (("edge-train-ppo", flat), ("edge-train-bc", rising)):
        for seed in (0, 1):
            f = p2 / slug / ("%s_seed%d.pt.history.json" % (slug.split("-")[-1], seed))
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(hist(vals)), encoding="utf-8")
    hs = ro.load_histories(p2)
    rows = ro.curve_rows(hs)
    assert len(rows) == 4 and ro.unconverged_learners(rows) == ["edge-train-bc"]
    ctx = ro.Ctx(tmp_path / "o", scratch=tmp_path)
    ctx.p2 = p2
    ro.analyse_curves(ctx)
    assert (ctx.out / "curves.csv").exists() and (ctx.out / "curves" / "edge-train-ppo.png").exists()


def test_tuned_pick_and_checkpoints(tmp_path):
    def shard(best_v, ctr):
        return dict(sv_dpp_grid=[dict(kwargs=dict(V=0.1, hard_purge_prob=0.9), ctr_ungated=0.2),
                                 dict(kwargs=dict(V=best_v, hard_purge_prob=0.95), ctr_ungated=ctr)])
    p2 = tmp_path / "p2"
    for slug, i, d in (("edge-sweep-tune-0of2", 0, shard(1.0, 0.5)), ("edge-sweep-tune-1of2", 1, shard(3.0, 0.7)),
                       ("edge-sweep-tune-0of2-fusion", 0, shard(10.0, 0.6))):
        f = p2 / slug / ("tune.shard%dof2.json" % i)
        f.parent.mkdir(parents=True)
        f.write_text(json.dumps(d), encoding="utf-8")
    res = ro.pick_tuned(ro.tune_files(p2))
    assert res["nofusion"]["best_kwargs"]["V"] == 3.0 and res["nofusion"]["n_shards"] == 2
    assert res["fusion"]["best_kwargs"]["V"] == 10.0
    for slug, name in (("edge-train-scope-ppo", "scope_ppo_seed0.pt"), ("edge-train-scope-ppo-fusion", "scope_ppo_seed1.pt"),
                       ("edge-train-ltr-ranknet", "ltr_seed0.pkl")):
        (p2 / slug).mkdir(parents=True, exist_ok=True)
        (p2 / slug / name).write_bytes(b"x")
        (p2 / slug / (name + ".history.json")).write_text("[]")
    ck = ro.learned_checkpoints(p2)
    assert {(c["slug"], c["algo"], c["seed"]) for c in ck} == {
        ("edge-train-scope-ppo", "scope_ppo", 0), ("edge-train-scope-ppo-fusion", "scope_ppo", 1),
        ("edge-train-ltr-ranknet", "ltr", 0)}
    assert len(ro.seed0(ck)) == 2


def cell(variant, w, ctr, lo, hi, policy="sv_dpp", rho=1.2):
    return dict(policy=policy, variant=variant, workers=w, rho=rho, ctr_ungated=ctr, ci95=dict(ctr_ungated=[lo, hi]))


def test_header_verdict_pass_and_fail():
    good = [cell("cum:full", w, 0.7, 0.66, 0.74) for w in (1, 2)] + [cell("cum:spi", w, 0.6, 0.56, 0.64) for w in (1, 2)] \
        + [cell("cum:none", w, 0.5, 0.46, 0.54) for w in (1, 2)]
    ok, text = ro.header_verdict(good)
    assert ok and "c=2" in text
    bad = [dict(c) for c in good]
    bad[4] = cell("cum:none", 1, 0.8, 0.76, 0.84)                    # no-header beats full (the old failing result)
    ok, text = ro.header_verdict(bad)
    assert not ok and "no" in text
    overlap = good[:2] + [cell("cum:spi", w, 0.69, 0.60, 0.74) for w in (1, 2)] + good[4:]
    assert not ro.header_verdict(overlap)[0]
    assert not ro.header_verdict([])[0]


def test_merge_runs_and_review_sample(tmp_path):
    def run(cells):
        return dict(config=dict(split="val", policies=[], workers=[], rhos=[]), cells=cells)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(run([dict(policy="edf", workers=1, rho=1.0, v=1)])))
    b.write_text(json.dumps(run([dict(policy="fifo", workers=1, rho=1.0, v=2), dict(policy="edf", workers=1, rho=1.0, v=3)])))
    m = ro.merge_runs([a, b])
    assert sorted(c["policy"] for c in m["cells"]) == ["edf", "fifo"] and m["config"]["policies"] == ["edf", "fifo"]
    assert [c["v"] for c in m["cells"] if c["policy"] == "edf"] == [3]
    q = [dict(id=i, image_id="x:%d.jpg" % i, true_tier=i % 4, age=1.0, slack_hard=5.0, spi=[.25] * 4) for i in range(5)]
    t = tmp_path / "t.jsonl"
    t.write_text("\n".join(json.dumps(dict(kind="decision", t=float(k), queue_size=5, queue=q, batch=[k % 5]))
                           for k in range(50)) + "\n" + json.dumps(dict(kind="complete")) + "\n")
    s = ro.review_sample(t, lambda i: None)
    assert len(s) == 30 and s[0]["picked"][0]["true_tier"] is not None and s[0]["skipped_total"] == 4


def test_kaggle_plan_shards_both_accounts(tmp_path):
    ctx = ro.Ctx(tmp_path, dry=True, scratch=tmp_path)
    ck = [dict(slug="edge-train-scope-ppo", variant="nofusion", algo="scope_ppo", seed=0, path="/x/scope_ppo_seed0.pt"),
          dict(slug="edge-train-ppo", variant="nofusion", algo="ppo", seed=0, path="/x/ppo_seed0.pt")]
    plan = ro.split_names(ctx, ck)
    assert "learned:ppo:/x/ppo_seed0.pt" in plan["nofusion"]["local"] and plan["nofusion"]["learned"][0]["algo"] == "scope_ppo"
    nbs = ro.kaggle_notebooks(ctx, plan, "compare", {"edge-train-scope-ppo": "win"})
    assert {n[1] for n in nbs} == {"wsl", "win"} and len(nbs) == 4
    win = [n for n in nbs if n[1] == "win" and n[2] == "nofusion"][0]
    assert "learned:scope_ppo:scope_ppo_seed0.pt" in win[3] and win[4] == ["edge-train-scope-ppo"]

"""Qwen feedback entry point: sampling, root discovery, plan, manifest (no model, no GPU)."""
import json
from collections import Counter

from scripts.edge import make_qwen_feedback_notebook as mk
from Scheduler.training.edge import qwen_feedback as q


def _rows():
    rows = []
    for t, n in (("Critical", 30), ("High", 40), ("Medium", 50), ("Low", 200)):
        for k in range(n):
            rows.append({"uid": "%s%d" % (t[0], k), "file": "images/x/%s%d.jpg" % (t[0], k), "tier": t,
                         "source": "crisismmd" if k % 2 else "places365", "event": "harvey", "split": "train"})
    rows += [dict(r, uid="v" + r["uid"], split="val") for r in rows[:40]]
    return rows


def test_stratified_sample_balanced_and_deterministic():
    rows = [r for r in _rows() if r["split"] == "train"]
    s = q.stratified_sample(rows, 100, 0)
    assert len(s) == len({r["uid"] for r in s}) == 100
    assert all(c == 25 for c in Counter(r["tier"] for r in s).values())
    assert {r["uid"] for r in s} == {r["uid"] for r in q.stratified_sample(rows, 100, 0)}
    assert len(q.stratified_sample(rows, 10_000, 0)) == len(rows)


def test_pairs_span_tier_differences_and_are_distinct():
    rows = [r for r in _rows() if r["split"] == "train"]
    ps = q.sample_pairs(rows, 200, 1)
    assert len(ps) == 200 and len({frozenset((a["uid"], b["uid"])) for a, b in ps}) == 200
    rank = {t: i for i, t in enumerate(q.TIERS)}
    assert {abs(rank[a["tier"]] - rank[b["tier"]]) for a, b in ps} >= {1, 2, 3}


def test_size_plan_fits_budget():
    s = q.size_plan(10, 1.5, 4.0)
    assert (s["val"] * 1.5 + s["point"] * 1.5 + s["pair"] * 4.0) / 2 <= 10 * 3600 + 1


def test_plan_uses_train_for_feedback_and_val_only_for_audit():
    plan = q.build_plan(_rows(), {"val": 20, "point": 40, "pair": 10})
    assert all(r["uid"][0] != "v" for r in plan["point"]) and all(r["uid"][0] == "v" for r in plan["val"])
    assert all(a["uid"][0] != "v" and b["uid"][0] != "v" for a, b in plan["pair"])


def test_discover_and_load(tmp_path):
    a, b = tmp_path / "datasets/o/img", tmp_path / "datasets/o/lab"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "index.jsonl").write_text("")
    rows = [{"uid": "u1", "file": "f", "tier": "Low", "source": "s"}]
    (b / "labels_raw.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (b / "splits_v1.json").write_text(json.dumps({"splits": {"u1": "val"}}))
    assert q.discover_inputs(tmp_path) == (a, b)
    assert q.load_rows(b)[0]["split"] == "val"


def test_summary_and_manifest(tmp_path):
    s = q.summarize({"point_train": [{"label": 1, "logp": [0] * 4}, {}],
                     "pair_train": [{"consistent": True}, {"consistent": False}]})
    assert s["point_train"]["parse_rate"] == 0.5 and s["pair_train"]["consistent_rate"] == 0.5
    p = q.write_manifest(tmp_path / "o", {"model_id": "m", "git_sha": "abc"})
    assert json.loads(p.read_text())["model_id"] == "m"


def test_notebook_builder_is_thin():
    nb, meta = mk.build("pilot")
    assert meta["id"] == "satyapaladuguplus/edge-qwen-feedback" and meta["enable_internet"] and meta["enable_gpu"]
    assert set(meta["dataset_sources"]) == set(mk.DATASET_SOURCES)
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert "Scheduler.training.edge.qwen_feedback" in src and "--mode" in src

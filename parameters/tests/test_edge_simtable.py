import json

import numpy as np

from parameters.edge import simtable as st

MISSIONS = {"model": "x", "missions": [{"id": 0, "name": "general"}, {"id": 1, "name": "flood"}],
            "event_mission": {"e_tr": "flood", "e_va": "flood", "e_te": "flood"}, "everyday_mission": "general"}


def test_tweet_time_from_snowflake():
    import datetime
    t = datetime.datetime.fromtimestamp(st.tweet_time_s(917791044158185473), datetime.timezone.utc)
    assert t.date() == datetime.date(2017, 10, 10)          # CrisisMMD folder california_wildfires/10_10_2017
    assert st.tweet_time_s(None) is None


def test_relevance_matrix_range_and_order():
    rel = st.relevance_matrix([[1, 0], [0, 1]], [[1, 0], [-1, 0]])
    assert np.allclose(rel, [[1.0, 0.0], [0.5, 0.5]])


def test_dup_groups_connected_components():
    sem = np.array([0, 1, 0xFFFF << 30, 0xFFFF << 40], np.uint64)        # 0~1 by SimHash
    per = np.array([0, 0xFFFFFFFF, 0xFFFFF000, 0xFFFFF000], np.uint64)    # 2~3 by pHash (distance 0); far from 0, 1
    g = st.dup_groups(sem, per, sem_max=1, perc_max=8)
    assert g[0] == g[1] != -1 and g[2] == g[3] != -1 and g[0] != g[2]
    alone = st.dup_groups(np.array([0, 0xFFFF], np.uint64), np.array([0, 0xFFFF], np.uint64), 1, 8)
    assert (alone == -1).all()


def _fake(n_per=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, splits, emb = [], {}, []
    for s, ev in (("train", "e_tr"), ("val", "e_va"), ("test", "e_te")):
        for t, hum in enumerate(["injured_or_dead_people", "vehicle_damage", "affected_individuals", "not_humanitarian"]):
            for k in range(n_per):
                uid = "crisismmd:%s_%d_%d" % (s, t, k)
                rows.append({"uid": uid, "source": "crisismmd", "event": ev, "tier": "Low", "tweet_id": str(10 ** 18 + k),
                             "damage": [], "humanitarian": [hum], "informative": ["informative"], "flags": []})
                splits[uid] = s
                e = rng.normal(size=8); e[t] += 6.0
                emb.append(e / np.linalg.norm(e))
    pos = {r["uid"]: i for i, r in enumerate(rows)}
    sem = rng.integers(0, 2 ** 63, size=len(rows), dtype=np.uint64)
    per = rng.integers(0, 2 ** 63, size=len(rows), dtype=np.uint64)
    return rows, splits, (pos, np.array(emb, np.float32), sem, per)


def test_build_rows_validate_and_follow_v3_mapping():
    rows, splits, cache = _fake()
    out, head, rep = st.build(rows, splits, cache, np.eye(2, 8), MISSIONS, {"simhash_max": 1, "phash_max": 8})
    assert len(out) == len(rows)
    by = {r["image_id"]: r for r in out}
    assert by["crisismmd:train_2_0"]["tier"] == 2 and by["crisismmd:train_2_0"]["tier_alt"] == 1   # affected: Medium / alt High
    assert all(r["fidelity"] == 1 and r["sts"] == [] and r["mission"] == 1 for r in out)
    assert all(len(r["relevance_by_mission"]) == 2 and r["relevance"] == r["relevance_by_mission"][1] for r in out)
    assert rep["spi"]["test_scored"] is False and rep["spi"]["val"]["n"] == 48


def _c2(tmp_path, ids, bad=()):
    """A feature-pass cache folder (one shard, m 0.25 and 0.55) for ``ids``."""
    sh = tmp_path / "c2" / "shards" / "shard_0000"
    sh.mkdir(parents=True)
    n = len(ids)
    (sh / "uids.json").write_text(json.dumps(ids))
    np.save(sh / "sqgan_ok.npy", np.array([[i not in bad, i not in bad] for i in range(n)]))
    for tag, size in (("m0.25", 100), ("m0.55", 220)):
        np.save(sh / ("offsets_%s.npy" % tag), np.arange(n + 1, dtype=np.int64) * size)
    np.save(sh / "sts_m0.25.npy", np.tile(np.linspace(0, 1, 13, dtype=np.float32), (n, 1)))
    np.save(sh / "lpips_m0.25.npy", np.full(n, 0.12, np.float32))
    np.save(sh / "miou_m0.25.npy", np.full(n, 0.8, np.float32))
    (sh / "latency_samples.json").write_text(json.dumps({"samples": {"1": [0.01, 0.02], "2": [0.03]}}))
    return tmp_path / "c2"


def _table():
    rows, splits, cache = _fake()
    return st.build(rows, splits, cache, np.eye(2, 8), MISSIONS, {"simhash_max": 1, "phash_max": 8})[0]


def test_fill_codec_merges_by_image_id_and_reports_missing(tmp_path):
    from Evaluation.edge.tables import latency_from_profile_json, validate_row
    table = _table()
    ids = [r["image_id"] for r in table]
    c2, samples = st.load_c2(_c2(tmp_path, ids[:-2] + ["not-in-table"], bad=(0,)))
    out, rep, lat = st.fill_codec(table, c2, samples)
    by = {r["image_id"]: r for r in out}
    f = by[ids[1]]
    assert f["payload_source"] == "sqgan_v1" and f["payload_bytes"] == 100 and f["payload_bytes_by_m"] == {"0.25": 100, "0.55": 220}
    assert len(f["sts"]) == 13 and abs(f["lpips"] - 0.12) < 1e-6 and abs(f["recon_miou"] - 0.8) < 1e-6
    assert by[ids[0]]["payload_source"] == "placeholder" and by[ids[0]]["sts"] == []     # C2 flagged image 0 as failed
    assert rep["filled"] == len(ids) - 3 and rep["missing_from_c2"] == 3 and rep["c2_not_in_table"] == 1
    assert all(validate_row(dict(r)) for r in out)
    p = tmp_path / "latency_v1.json"
    p.write_text(json.dumps(lat))
    assert latency_from_profile_json(p) is not None and lat["samples"]["1"] == [0.01, 0.02]


def test_fill_codec_cli_writes_outputs(tmp_path):
    table = _table()
    tpath = tmp_path / "simtable_v1.jsonl"
    tpath.write_text("\n".join(json.dumps(r) for r in table) + "\n")
    rep = st.main(["fill-codec", "--table", str(tpath), "--c2", str(_c2(tmp_path, [r["image_id"] for r in table])),
                   "--out", str(tmp_path / "o")])
    assert rep["filled"] == len(table)
    assert json.loads((tmp_path / "o" / "latency_v1.json").read_text())["samples"]["2"] == [0.03]
    assert (tmp_path / "o" / "simtable_v1.jsonl").exists() and (tmp_path / "o" / "codec_fill_report.json").exists()

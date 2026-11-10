import json

import numpy as np
import pytest

from Evaluation.edge import tables


def row(i, **kw):
    r = {"image_id": i, "event": "e", "source": "s", "tier": 1, "tier_probs": [0.1, 0.6, 0.2, 0.1],
         "relevance": 0.3, "sem_key": 5, "perc_key": 7, "sts": [0.0] * 13, "payload_bytes": 100,
         "fidelity": 1, "dup_group": None, "mission": 0, "codec": "v0", "split": "test"}
    r.update(kw)
    return r


def write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_merge(tmp_path):
    a, b, c = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    write(a, [{"image_id": "x", "tier": 1, "sts": None}])
    write(b, [{"image_id": "x", "sts": [1.0], "tier": 1}, {"image_id": "y", "tier": 2}])
    merged = {r["image_id"]: r for r in tables.merge_partial_tables([a, b])}
    assert merged["x"]["sts"] == [1.0] and set(merged) == {"x", "y"}
    write(c, [{"image_id": "x", "tier": 3}])
    with pytest.raises(ValueError):
        tables.merge_partial_tables([a, c])


def test_validate_row():
    tables.validate_row(row(0))
    tables.validate_row(row(0, sts=[]))
    for bad in (dict(tier=4), dict(tier_probs=[0.5, 0.5, 0.5, 0.5]), dict(sts=[0.0] * 5),
                dict(codec="png"), dict(split="dev"), dict(fidelity=2), dict(sem_key=-1)):
        with pytest.raises(ValueError):
            tables.validate_row(row(0, **bad))
    r = row(0)
    del r["mission"]
    with pytest.raises(ValueError):
        tables.validate_row(r)


def test_jobs_and_profile(tmp_path):
    rows = [row(i, split="test" if i < 8 else "val") for i in range(10)]
    test = tables.load_split(rows, "test")
    assert len(test) == 8
    jobs = tables.rows_to_jobs(test, [3.0, 1.0, 2.0, 5.0], np.random.default_rng(0), network_delay=0.1)
    arr = [j.header.arrival_time for j in jobs]
    assert arr == sorted(arr) and len({j.id for j in jobs}) == 4
    assert abs(jobs[0].header.capture_time - 0.9) < 1e-9 and jobs[0].truth.tier == 1
    with pytest.raises(ValueError):
        tables.rows_to_jobs(test, list(range(9)), np.random.default_rng(0))
    p = tmp_path / "lat.json"
    p.write_text(json.dumps({"1": [0.1, 0.2], "4": [0.3, 0.4]}))
    assert tables.latency_from_profile_json(p).batch_sizes == (1, 4)

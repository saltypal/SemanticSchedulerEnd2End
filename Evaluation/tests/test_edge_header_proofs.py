"""Header-proof runner (I9): drop-out masks, equal-bytes, dedup P/R, overhead, CSV/JSON."""
import csv
import json

import numpy as np
import pytest

from Evaluation.edge import header_proofs as hp
from Evaluation.edge import tables
from Evaluation.edge.contracts import Header
from Evaluation.tests.test_edge_scenarios import LAT, make_rows

H = Header(1.0, 2.0, 464, 1, 1, 0, (0.7, 0.1, 0.1, 0.1), 0.62, 12345, 67890, tuple([0.3] * 13))


def test_neutralise_masks_exactly_the_named_fields():
    h = hp.neutralise(H, ("spi",), 7)
    assert h.tier_probs == hp.UNIFORM and h.relevance == H.relevance and h.sem_key == H.sem_key
    h = hp.neutralise(H, ("relevance", "sts"), 7)
    assert h.relevance == hp.CONST_RELEVANCE and h.sts == () and h.tier_probs == H.tier_probs
    h = hp.neutralise(H, ("keys",), 7)
    assert (h.sem_key, h.perc_key) != (H.sem_key, H.perc_key)
    h2 = hp.neutralise(H, ("keys",), 7)
    assert (h.sem_key, h.perc_key) == (h2.sem_key, h2.perc_key)
    assert h.capture_time == H.capture_time and h.payload_bytes == H.payload_bytes


def test_variants_cover_cumulative_and_leave_one_out():
    v = hp.variants()
    assert v["cum:full"] is None
    assert [k for k in v if k.startswith("cum:")] == ["cum:none", "cum:spi", "cum:spi+relevance",
                                                     "cum:spi+relevance+keys", "cum:full"]
    assert v["cum:none"](H, 1).tier_probs == hp.UNIFORM and v["cum:none"](H, 1).sts == ()
    assert v["cum:spi"](H, 1).tier_probs == H.tier_probs
    assert v["cum:spi"](H, 1).relevance == hp.CONST_RELEVANCE
    assert v["drop:keys"](H, 1).sem_key != H.sem_key and v["drop:keys"](H, 1).sts == H.sts
    eb = v["equal_bytes"](H, 1)
    assert eb.tier_probs == hp.UNIFORM and len(eb.sts) == 13 and eb.payload_bytes == H.payload_bytes
    assert eb.sts != H.sts


def test_dedup_pr_and_overhead():
    rows = [dict(r, dup_group=i // 2, sem_key=(i // 2) * 0x9E3779B97F4A7C15 % 2 ** 64,
                 perc_key=(i // 2) * 0xC2B2AE3D27D4EB4F % 2 ** 64)
            for i, r in enumerate(make_rows(40, 40))]
    jobs = tables.rows_to_jobs_ordered(rows, np.arange(40.0))
    pr = hp.dedup_pr(jobs)[0.5]
    assert pr["recall"] == 1.0 and pr["n_true_dups"] == 20 and pr["precision"] > 0.9
    assert hp.overhead(rows)["mean"] == pytest.approx(64 / (64 + 464))


def test_end_to_end_csv_json(tmp_path):
    res = hp.run_header_proofs(make_rows(), LAT, "val", ["fifo"], [1], [1.0], 1, 0, tables,
                               arrival="trace", n_boot=5, max_batch=2)
    names = {c["variant"] for c in res["ablation"]}
    assert {"cum:none", "cum:full", "drop:spi", "equal_bytes"} <= names
    hp.save(res, str(tmp_path))
    with open(tmp_path / "header_proofs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["experiment"] for r in rows} == {"ablation", "equal_bytes", "dedup", "overhead"}
    with open(tmp_path / "header_proofs.json", encoding="utf-8") as f:
        assert json.load(f)["overhead"]["n"] == 80

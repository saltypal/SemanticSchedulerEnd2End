"""D19 STS fusion: fit, belief, env wiring, sweep flag, header-proof drop:sts. Synthetic, CPU."""
import json

import numpy as np

from Evaluation.edge import header_proofs as hp
from Evaluation.edge import run_sweep, tables
from Evaluation.edge.contracts import Header
from Evaluation.edge.receiver import Receiver
from Evaluation.tests.test_edge_scenarios import LAT, make_rows
from parameters.edge import sts_fusion as sf


def toy_rows(n=600, seed=0):
    r = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        y = int(r.integers(0, 4))
        logit = r.normal(0, 1.0, 4)
        logit[y] += 0.3                                   # weak SPI
        p = np.exp(logit) / np.exp(logit).sum()
        sts = r.normal(0, 0.3, 13)
        sts[y] += 2.0                                     # STS carries the tier
        rows.append(dict(tier=y, tier_probs=p.tolist(), sts=sts.tolist(),
                         split="train" if i < 400 else "val"))
    return rows


def test_fit_improves_val_over_spi_and_roundtrips():
    rows = toy_rows()
    fus = sf.fit(rows)
    rep = sf.report(rows, fus)
    assert rep["delta"]["macro_f1"] > 0.1 and rep["n_val"] == 200
    fus2 = sf.StsFusion.from_dict(json.loads(json.dumps(fus.to_dict())))
    r = rows[450]
    assert np.allclose(fus(r["tier_probs"], r["sts"]), fus2(r["tier_probs"], r["sts"]))
    assert abs(sum(fus(r["tier_probs"], r["sts"])) - 1) < 1e-9


H = Header(1.0, 2.0, 464, 1, 1, 0, (0.7, 0.1, 0.1, 0.1), 0.62, 5, 6, tuple([0.3] * 13))


def _fusion():
    return sf.fit(toy_rows())


def test_belief_unchanged_without_sts_or_fusion():
    import dataclasses
    rx = Receiver(fusion=_fusion())
    h0 = dataclasses.replace(H, sts=())
    assert rx.belief(h0) is h0 and Receiver().belief(H) is H
    assert rx.belief(H).tier_probs != H.tier_probs and rx.belief(H).sem_key == H.sem_key


def test_drop_sts_equals_spi_only_belief():
    rx = Receiver(fusion=_fusion())
    h = hp.variants()["drop:sts"](H, 1)
    assert rx.belief(h).tier_probs == H.tier_probs


def test_env_policy_sees_belief_metrics_use_truth():
    from Evaluation.edge.env import EdgeEnv
    seen = []

    class Spy(Receiver):
        def belief(self, header):
            seen.append(header)
            return super().belief(header)

    rows = make_rows()
    for r in rows:
        r["sts"] = [0.5] * 13
    jobs = tables.rows_to_jobs(rows[:20], np.arange(20) * 1.0, np.random.default_rng(0))
    spec, mb = run_sweep._setup(rows, LAT, "val", None, 2)[1:]
    env = EdgeEnv(jobs, LAT, spec, 1, mb, Spy(fusion=_fusion()), 0)
    assert env._belief_header(jobs[0]).tier_probs != jobs[0].header.tier_probs
    env._belief_header(jobs[0])
    assert len(seen) == 1                                  # cached per job
    assert len(jobs[0].header.sts) == 13 and jobs[0].header.tier_probs != env._belief_header(jobs[0]).tier_probs


def test_sweep_with_and_without_fusion(tmp_path):
    rows = make_rows()
    for r in rows:
        r["sts"] = [0.2 * (r["tier"] + 1)] * 13
    base = dict(rows=rows, latency=LAT, split="val", policies=["fifo"], workers=[1], rhos=[1.0],
                realisations=1, base_seed=1, max_batch=2, n_boot=5, tables=tables)
    a = run_sweep.run_sweep(**base)
    b = run_sweep.run_sweep(**base, sts_fusion=_fusion())
    assert len(a["cells"]) == len(b["cells"]) == 1
    p = tmp_path / "f.json"
    p.write_text(json.dumps(_fusion().to_dict()))
    assert run_sweep._load_fusion(str(p)) is not None and run_sweep._load_fusion(None) is None

"""Live decoder validation (plan v3 sec. 9): real decode latency vs the simulator, c = 2 workers.

A live queue is fed by the *same arrival trace* as ``EdgeEnv``; each policy drives ``workers`` worker
threads (one per GPU) that call an injectable ``decode_fn(worker, jobs) -> None`` (blocking). The
measured wall time of each call is the live latency. Reports per batch size a two-sample KS test of
live vs profile latency, and per policy CTR@D live vs simulated with a paired bootstrap CI of the
difference. The real SQ-GAN decoder is built lazily in ``make_sqgan_decoder``; tests use fakes.
CLI: ``python -m Evaluation.edge.live_check --table T --profile P --out O --decoder sqgan|fake``.
"""
import argparse
import json
import math
import queue
import threading
import time
from math import exp

import numpy as np

from Evaluation.edge import metrics
from Evaluation.edge.contracts import QueuedJob, State, ValueSpec
from Evaluation.edge.env import RATE_TAU, EdgeEnv
from Evaluation.edge.receiver import Receiver

DEFAULT_POLICIES = ("fifo", "edf", "sv_dpp")


def live_run(jobs, policy, decode_fn, latency, spec, workers=2, max_batch=4, seed=0,
             clock=time.perf_counter, receiver=None):
    """Run ``policy`` live. Returns ``(outcome rows like EdgeEnv, decode_log)``.

    decode_log: one dict per decode call: worker, batch_size, latency_s, start, ids.
    Trace times are seconds since the run start (real time, scale 1).
    """
    jobs = sorted(jobs, key=lambda j: (j.header.arrival_time, j.id))
    receiver = receiver or Receiver()
    policy.reset(spec, np.random.default_rng(seed + 1))
    h_max = max(spec.hard_deadline(k) for k in range(len(spec.weights)))
    events, inboxes, log, errors = queue.Queue(), [queue.Queue() for _ in range(workers)], [], []

    def worker_loop(w):
        while True:
            batch = inboxes[w].get()
            if batch is None:
                return
            t0 = clock()
            try:
                decode_fn(w, list(batch))
            except BaseException as exc:                    # surface in the dispatcher
                errors.append(exc)
            events.put((w, batch, t0, clock() - t0))

    threads = [threading.Thread(target=worker_loop, args=(w,), daemon=True) for w in range(workers)]
    for t in threads:
        t.start()

    start = clock()
    now_fn = lambda: clock() - start
    qd, rows, delivered, mission_val = {}, {}, set(), {}
    free, ai, inflight = list(range(workers)), 0, 0
    rate, last_arr = 0.0, None

    def queued(job, now):
        h = job.header
        return QueuedJob(job.id, h, now - h.capture_time, receiver.dup_prob(h), receiver.u_hat(h))

    try:
        while ai < len(jobs) or inflight:
            now = now_fn()
            nxt = jobs[ai].header.arrival_time if ai < len(jobs) else None
            timeout = None if nxt is None else max(0.0, nxt - now)
            try:
                w, batch, t0, dur = events.get(timeout=timeout)
                inflight -= 1
                free.append(w)
                if errors:
                    raise errors[0]
                now = now_fn()
                log.append(dict(worker=w, batch_size=len(batch), latency_s=dur, start=t0 - start,
                                ids=[j.id for j in batch]))
                done = {"ids": [], "tier_probs": [], "delays": []}
                for job in batch:
                    k, d, grp = job.truth.tier, now - job.header.capture_time, job.truth.dup_group
                    dup = grp is not None and grp in delivered
                    zeta = (spec.zeta_critical_copy if k == 0 else 0.0) if dup else 1.0
                    rows[job.id] = EdgeEnv._row(job, "decoded", now, t0 - start, len(batch),
                                                spec.weights[k] * spec.g(k, d) * zeta, dup)
                    if grp is not None:
                        delivered.add(grp)
                    receiver.deliver(job.header)
                    mission_val[job.header.mission] = mission_val.get(job.header.mission, 0.0) + sum(
                        p * wt for p, wt in zip(job.header.tier_probs, spec.weights))
                    done["ids"].append(job.id)
                    done["tier_probs"].append(job.header.tier_probs)
                    done["delays"].append(d)
                policy.observe("complete", dict(done, time=now, batch_size=len(batch)))
            except queue.Empty:
                pass
            now = now_fn()
            while ai < len(jobs) and jobs[ai].header.arrival_time <= now:
                job = jobs[ai]
                ai += 1
                if last_arr is not None:
                    rate *= exp(-(now - last_arr) / RATE_TAU)
                rate += 1.0 / RATE_TAU
                last_arr = now
                qd[job.id] = job
                policy.observe("arrival", dict(id=job.id, time=now, tier_probs=job.header.tier_probs))
            for jid in [i for i, j in qd.items() if now - j.header.capture_time > h_max]:
                rows[jid] = EdgeEnv._row(qd.pop(jid), "expired", now, None, None)
                policy.observe("expire", dict(ids=[jid], time=now))
            while free and qd:
                now = now_fn()
                cur = rate * exp(-(now - last_arr) / RATE_TAU) if last_arr is not None else 0.0
                state = State(now, tuple(queued(j, now) for j in qd.values()), len(free), workers, latency,
                              spec, max_batch, arrival_rate=cur,
                              mission_served=tuple(sorted(mission_val.items())))
                dec = policy.select(state)
                for jid in dec.purge:
                    rows[jid] = EdgeEnv._row(qd.pop(jid), "purged", now, None, None)
                    policy.observe("purge", dict(ids=[jid], time=now))
                if not dec.batch:
                    if dec.purge:
                        continue
                    break
                if len(dec.batch) not in latency.batch_sizes or len(dec.batch) > max_batch:
                    raise ValueError("invalid batch size %d" % len(dec.batch))
                batch = [qd.pop(i) for i in dec.batch]
                inboxes[free.pop(0)].put(batch)
                inflight += 1
    finally:
        for q in inboxes:
            q.put(None)
        for t in threads:
            t.join(timeout=5)
    for jid, job in qd.items():
        rows[jid] = EdgeEnv._row(job, "queued_at_end", None, None, None)
    for r in rows.values():
        r["measured"] = True
    return list(rows.values()), log


def simulate(jobs, policy_factory, latency, spec, workers, max_batch, seeds):
    """One simulated outcome list per seed on the same jobs (no warm-up cut, to match live)."""
    out = []
    for s in seeds:
        env = EdgeEnv(jobs, latency, spec, workers, max_batch, Receiver(), int(s), warmup_fraction=0.0)
        out.append(env.run(policy_factory()).outcomes)
    return out


def _crit_indicator(rows, d0):
    """{job id: 1.0 if tier-0 job decoded within d0 else 0.0}."""
    return {r["id"]: float(r["status"] == "decoded" and r["delay"] is not None and r["delay"] <= d0)
            for r in rows if r["tier"] == 0}


def ks_per_batch(log, latency, min_n=5):
    """Two-sample KS of live decode latency vs the profile samples, per batch size."""
    from scipy.stats import ks_2samp
    out = {}
    for b in latency.batch_sizes:
        live = np.array([e["latency_s"] for e in log if e["batch_size"] == b], dtype=float)
        ref = latency.samples[b]
        entry = dict(n_live=int(len(live)), n_ref=int(len(ref)), ref_median_s=float(np.median(ref)))
        if len(live) >= min_n:
            res = ks_2samp(live, ref)
            entry.update(statistic=float(res.statistic), p_value=float(res.pvalue),
                         live_median_s=float(np.median(live)))
        else:
            entry.update(statistic=None, p_value=None, live_median_s=float(np.median(live)) if len(live) else None)
        out[str(b)] = entry
    return out


def ctr_delta(live_rows, sim_rows_list, d0, n_boot, rng):
    """CTR@D live and simulated plus a paired bootstrap CI (over tier-0 jobs) of live - sim."""
    live = _crit_indicator(live_rows, d0)
    sims = [_crit_indicator(r, d0) for r in sim_rows_list]
    ids = sorted(live)
    if not ids:
        return dict(n_critical=0, ctr_live=None, ctr_sim=None, delta=None, delta_ci95=[None, None])
    lv = np.array([live[i] for i in ids])
    sv = np.mean([[s.get(i, 0.0) for i in ids] for s in sims], axis=0)
    diff = lv - sv
    boots = [diff[rng.integers(len(ids), size=len(ids))].mean() for _ in range(int(n_boot))]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    lci = metrics.bootstrap_ci(ids, lambda s: float(np.mean([live[i] for i in s])), n_boot, rng)
    sci = metrics.bootstrap_ci(ids, lambda s: float(np.mean([sv[ids.index(i)] for i in s])), min(n_boot, 200), rng)
    return dict(n_critical=len(ids), ctr_live=float(lv.mean()), ctr_live_ci95=list(lci),
                ctr_sim=float(sv.mean()), ctr_sim_ci95=list(sci), delta=float(diff.mean()),
                delta_ci95=[float(lo), float(hi)])


def run_live_check(jobs, latency, spec, decode_fn, policy_factories, workers=2, max_batch=4,
                   sim_seeds=range(10), n_boot=1000, seed=0):
    """Full report: pooled live decode log -> KS per batch size; per-policy CTR@D comparison."""
    rng = np.random.default_rng(seed)
    d0, full_log, per_policy = spec.soft_deadline[0], [], {}
    for name, factory in policy_factories.items():
        live_rows, log = live_run(jobs, factory(), decode_fn, latency, spec, workers, max_batch, seed)
        full_log.extend(log)
        sim = simulate(jobs, factory, latency, spec, workers, max_batch, sim_seeds)
        per_policy[name] = dict(ctr_at_d=ctr_delta(live_rows, sim, d0, n_boot, rng),
                                n_decoded_live=sum(r["status"] == "decoded" for r in live_rows),
                                n_jobs=len(live_rows), n_decode_calls=len(log))
    return dict(workers=workers, max_batch=max_batch, n_jobs=len(jobs), d0_s=d0,
                ks_by_batch=ks_per_batch(full_log, latency), policies=per_policy,
                claim="valid for c = %d only" % workers)


# ---------------------------------------------------------------- real decoder and CLI
def fake_decoder(latency, seed=0):
    """Decoder that sleeps a profile-sampled time (smoke runs without GPU/model)."""
    rng, lock = np.random.default_rng(seed), threading.Lock()

    def decode(worker, batch):
        with lock:
            s = latency.sample(len(batch), rng)
        time.sleep(s)
    return decode


def make_sqgan_decoder(rows_by_id, images_dir, devices):
    """Real decoder: SQ-GAN v1 bundle, one codec per device. Payloads are pre-encoded (not timed)."""
    from pathlib import Path

    from PIL import Image

    from Codec.edge import feature_pass_backends as real
    from Evaluation.edge.edge_config import load_config
    cfg = load_config("feature_pass")
    backends = [real.RealSQGAN(cfg, d) for d in devices]
    m_levels = cfg["sqgan"]["m_levels"]
    index = {p.stem: p for p in Path(images_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    payloads = {}

    def payload_for(job):
        if job.id not in payloads:
            row = rows_by_id[job.id]
            arr = np.asarray(Image.open(index[str(row["image_id"])]).convert("RGB"))
            _, handle = backends[0].segment(arr)
            m = m_levels[int(row.get("m_index", 0))]
            payloads[job.id] = (backends[0].encode(arr, handle, m)["payload"], m)
        return payloads[job.id]

    def decode(worker, batch):
        for job in batch:
            payload, m = payload_for(job)
            backends[worker].decode(payload, m)
        real.cuda_sync(devices[worker])
    decode.prepare = lambda jobs: [payload_for(j) for j in jobs]
    return decode


def main(argv=None):
    from Evaluation.edge import arrivals, tables
    from Evaluation.edge.edge_config import load_config
    from Evaluation.edge.run_sweep import _make_jobs, git_sha, save_results
    from Scheduler.edge import registry
    ap = argparse.ArgumentParser(description="Live decode validation vs simulator (c = workers)")
    ap.add_argument("--table", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-images", type=int, default=1000)
    ap.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--max-batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--decoder", choices=["sqgan", "fake"], default="sqgan")
    ap.add_argument("--images-dir", default="/kaggle/input")
    ap.add_argument("--devices", nargs="+", default=None, help="default cuda:0 cuda:1 ...")
    a = ap.parse_args(argv)
    latency = tables.latency_from_profile_json(a.profile)
    spec = ValueSpec.from_config(load_config("sim"))
    rows = [r for r in tables.load_rows(a.table) if r.get("split") == a.split][: a.n_images]
    jobs = _make_jobs(rows, latency, tables, a.workers, a.rho, 0, a.seed, a.max_batch, None, 0.0)
    if a.decoder == "fake":
        decode = fake_decoder(latency, a.seed)
    else:
        devices = a.devices or ["cuda:%d" % i for i in range(a.workers)]
        decode = make_sqgan_decoder({i: r for i, r in enumerate(rows)}, a.images_dir, devices)
        decode.prepare(jobs)
    factories = {p: (lambda p=p: registry.build(p)) for p in a.policies}
    report = run_live_check(jobs, latency, spec, decode, factories, a.workers, a.max_batch,
                            n_boot=a.n_boot, seed=a.seed)
    report.update(git_sha=git_sha(), decoder=a.decoder, rho=a.rho, split=a.split)
    save_results(report, a.out)
    print(json.dumps({k: report[k] for k in ("ks_by_batch", "policies")}, indent=1, default=str)[:4000])
    return report


if __name__ == "__main__":
    main()

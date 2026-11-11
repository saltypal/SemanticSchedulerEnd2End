"""S6 bit-depth test (plan v3 sec. 4 footnote): fewest bits per header field with < 0.5 % decision flips on val.

Counterfactual, not two separate runs (two runs diverge after the first difference and overstate flips): at every
decision epoch the PRIMARY policy decides on full-precision headers and drives the simulation; a SHADOW copy of the
same policy sees the same queue with the chosen fields re-quantised to ``bits`` and its decision is only recorded.
Both receive the same ``observe`` events (the shadow's arrival payload is re-quantised too).
flip rate = epochs where the decisions differ / all epochs.

Fields: ``spi`` (4 tier probabilities), ``relevance`` (1 byte), ``sts`` (13 values; only once the codec table exists).
Run after the SQ-GAN v1 pass so the latency profile and STS are real (checklist I7).
"""
import argparse
import dataclasses
import json

import numpy as np

from Evaluation.edge.contracts import Decision, Policy
from parameters.edge.header64 import dequantise, quantise

FIELDS = ("spi", "relevance", "sts")
TARGET = 0.005


def _snap(values, bits):
    return tuple(dequantise(quantise(values, bits), bits).tolist())


def requantise_fields(header, bits, fields):
    """Copy of a contract Header with only ``fields`` snapped to ``bits`` levels."""
    kw = {}
    if "spi" in fields:
        p = np.asarray(_snap(header.tier_probs, bits))
        kw["tier_probs"] = tuple((p / p.sum()).tolist()) if p.sum() > 0 else (0.25,) * 4
    if "relevance" in fields:
        kw["relevance"] = float(_snap([header.relevance], bits)[0])
    if "sts" in fields and header.sts:
        kw["sts"] = _snap(header.sts, bits)
    return dataclasses.replace(header, **kw)


class ShadowPolicy(Policy):
    """Primary decides and acts; shadow decides on re-quantised headers and is only compared."""
    name = "shadow"

    def __init__(self, primary, shadow, bits, fields):
        self.primary, self.shadow, self.bits, self.fields = primary, shadow, int(bits), tuple(fields)
        self.epochs = self.flips = 0

    def reset(self, spec, rng):
        super().reset(spec, rng)
        self.primary.reset(spec, np.random.default_rng(0))
        self.shadow.reset(spec, np.random.default_rng(0))      # same seed: differences come from the header only

    def _q_state(self, state):
        q = tuple(dataclasses.replace(j, header=requantise_fields(j.header, self.bits, self.fields))
                  for j in state.queue)
        return dataclasses.replace(state, queue=q)

    def select(self, state):
        d = self.primary.select(state)
        s = self.shadow.select(self._q_state(state))
        self.epochs += 1
        self.flips += (set(d.batch), set(d.purge)) != (set(s.batch), set(s.purge))
        return d if isinstance(d, Decision) else Decision()

    def observe(self, event, payload):
        self.primary.observe(event, payload)
        if event == "arrival" and "spi" in self.fields and "tier_probs" in payload:
            p = np.asarray(_snap(payload["tier_probs"], self.bits))
            payload = dict(payload, tier_probs=tuple((p / max(p.sum(), 1e-12)).tolist()))
        self.shadow.observe(event, payload)

    @property
    def flip_rate(self):
        return self.flips / self.epochs if self.epochs else float("nan")


def study(make_env, make_policy, bits_list, fields_list):
    """``make_env()`` -> fresh EdgeEnv on the same jobs; ``make_policy()`` -> fresh policy. Returns rows."""
    rows = []
    for fields in fields_list:
        for bits in bits_list:
            sp = ShadowPolicy(make_policy(), make_policy(), bits, fields)
            make_env().run(sp)
            rows.append({"fields": "+".join(fields), "bits": int(bits), "epochs": sp.epochs,
                         "flip_rate": sp.flip_rate, "passes": sp.flip_rate < TARGET})
    return rows


def smallest_passing(rows):
    """Per field set, the fewest bits whose flip rate is below the target (None if none pass)."""
    out = {}
    for r in sorted(rows, key=lambda r: r["bits"]):
        if r["passes"] and r["fields"] not in out:
            out[r["fields"]] = r["bits"]
    return out


def main(argv=None):
    from Evaluation.edge import arrivals, tables
    from Evaluation.edge.edge_config import load_config
    from Evaluation.edge.env import EdgeEnv
    from Evaluation.edge.receiver import Receiver
    from Evaluation.edge.contracts import ValueSpec
    from Scheduler.edge import registry
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--latency", required=True, help="latency_v1.json (LatencyProfile samples per batch size)")
    ap.add_argument("--policy", default="sv_dpp")
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-batch", type=int, default=4)
    ap.add_argument("--bits", default="2,3,4,5,6,8")
    ap.add_argument("--fields", default="spi;relevance;sts")
    ap.add_argument("--n", type=int, default=2000, help="val rows used")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = [r for r in tables.load_rows(a.table) if r["split"] == "val"]
    rng = np.random.default_rng(a.seed)
    rows = [rows[i] for i in rng.permutation(len(rows))[:a.n]]
    latency = tables.latency_from_profile_json(a.latency)
    spec = ValueSpec.from_config(load_config("sim")) if hasattr(ValueSpec, "from_config") else ValueSpec()
    b = max(x for x in latency.batch_sizes if x <= a.max_batch)
    times = arrivals.poisson_times(len(rows), arrivals.rate_for_rho(a.rho, latency, b, a.workers), rng)
    jobs = tables.rows_to_jobs(rows, times, np.random.default_rng(a.seed + 1))

    def make_env():
        return EdgeEnv(jobs, latency, spec, a.workers, a.max_batch, Receiver(), a.seed + 2)
    fields_list = [tuple(f.split("+")) for f in a.fields.split(";")]
    res = study(make_env, lambda: registry.make(a.policy), [int(x) for x in a.bits.split(",")], fields_list)
    out = {"policy": a.policy, "rho": a.rho, "workers": a.workers, "n_rows": len(rows), "target": TARGET,
           "rows": res, "smallest_passing_bits": smallest_passing(res)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["smallest_passing_bits"]))


if __name__ == "__main__":
    main()

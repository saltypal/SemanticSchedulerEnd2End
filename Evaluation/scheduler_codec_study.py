"""Run frozen FIFO, SPI and SQ01-cue policies on matched measured traces."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from Evaluation.scheduler_codec_replay import Observation, POLICIES, replay, trace


def cue_check(records, rows):
    eligible = [(r, o) for r, o in zip(records, rows)
                if r.get("event_group") and o.spi is not None and o.cue is not None]
    groups = sorted({r["event_group"] for r, _ in eligible})
    if len(groups) < 3:
        return dict(status="unqualified", reason="Need at least three event groups with SPI and encoder cue",
                    eligible=len(eligible), groups=len(groups))

    groups_by_hash = {}
    for record, _ in eligible:
        key = record.get("source_sha256")
        if not key:
            return dict(status="unqualified", reason="Missing source hash for group isolation")
        if groups_by_hash.setdefault(key, record["event_group"]) != record["event_group"]:
            return dict(status="unqualified", reason="Exact source image crosses event groups")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss

    group = np.array([r["event_group"] for r, _ in eligible])
    y = np.array([bool(o.predicted_positive) and o.reference_positive for _, o in eligible], dtype=int)
    spi = np.eye(4)[[o.spi for _, o in eligible]]
    cue = np.array([o.cue for _, o in eligible])
    x = np.column_stack((spi, cue))
    results = {}
    for name, design in (("spi", spi), ("spi_cue", x)):
        p = np.zeros(len(y))
        for value in groups:
            test = group == value
            if len(np.unique(y[~test])) != 2:
                return dict(status="unqualified", reason="A training fold has only one outcome class")
            model = LogisticRegression(random_state=1729).fit(design[~test], y[~test])
            p[test] = model.predict_proba(design[test])[:, 1]
        results[name] = dict(brier=float(brier_score_loss(y, p)),
                             average_precision=float(average_precision_score(y, p)))
    return dict(status="development_only", groups=len(groups), eligible=len(y), scores=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.receipt.read_bytes()
    receipt = json.loads(raw)
    records = receipt["rows"]
    rows = [Observation(**row["observation"]) for row in records]
    if receipt["codec"] == "jpeg" and any(row.cue is not None for row in rows):
        raise ValueError("JPEG control cannot carry the SQ01 mask cue")
    results = []
    for load in (0.5, 0.9, 1.2):
        for burst in (False, True):
            jobs, horizon = trace(rows, load, burst=burst)
            results.append(dict(nominal_load=load, burst=burst, horizon_ms=horizon,
                                offered_service_to_horizon=sum(o.decode_ms + o.reader_ms for o in rows) / horizon,
                                policies={name: replay(jobs, horizon, name) for name in POLICIES}))
    output = dict(scope="Development replay. Synthetic captures, four serial sender queues, ideal delivery, "
                        "one nonpreemptive receiver. Deadline six median service times from capture.",
                  codec=receipt["codec"], source_sha256=hashlib.sha256(raw).hexdigest(),
                  cue_bytes=receipt["cue_bytes"], cue_diagnostic=cue_check(records, rows),
                  holdout_opened=False, results=results)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

"""Run frozen FIFO, SPI and SQ01-cue policies on matched measured traces."""

import argparse
import hashlib
import json
import random
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
    parser.add_argument("--source-audit", type=Path, help="Publisher provenance for named events")
    parser.add_argument("--extension", type=Path, help="Additional image-disjoint pilot receipt")
    parser.add_argument("--cycles", type=int, default=1, help="Exact workload repetitions, not independent samples")
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    raw = args.receipt.read_bytes()
    receipt = json.loads(raw)
    records = receipt["rows"]
    extension_hash = None
    if args.extension:
        extension_raw = args.extension.read_bytes()
        extension = json.loads(extension_raw)
        for field in ("codec", "sqgan_checkpoint_sha256", "reader_checkpoint_sha256",
                      "preprocessing", "threshold", "cue_bytes"):
            if receipt.get(field) != extension.get(field):
                raise ValueError(f"pilot receipts use different {field}")
        if receipt.get("image_fraction", 0.25) != extension.get("image_fraction", 0.25):
            raise ValueError("pilot receipts use different image retention fractions")
        spi_keys = ("checkpoint_sha256", "head_sha256", "processor_sha256")
        if any(receipt.get("spi", {}).get(key) != extension.get("spi", {}).get(key)
               for key in spi_keys):
            raise ValueError("pilot receipts use different SPI identities")
        records = [*records, *extension["rows"]]
        if len({row["source_sha256"] for row in records}) != len(records):
            raise ValueError("pilot receipts contain duplicate images")
        if len({row["observation"]["identity"] for row in records}) != len(records):
            raise ValueError("pilot receipts contain duplicate identities")
        extension_hash = hashlib.sha256(extension_raw).hexdigest()
    source_audit_hash = None
    if args.source_audit:
        audit_raw = args.source_audit.read_bytes()
        matches = {row["identity"]: row for row in json.loads(audit_raw)["rows"]}
        for record in records:
            match = matches[record["observation"]["identity"]]
            if match["source_sha256"] != record["source_sha256"]:
                raise ValueError("source audit does not match the measured bytes")
            record["event_group"] = match["candidate_event"] if match["image_match"] else None
        source_audit_hash = hashlib.sha256(audit_raw).hexdigest()
    rows = [Observation(**row["observation"]) for row in records]
    if receipt["codec"] == "jpeg" and any(row.cue is not None for row in rows):
        raise ValueError("JPEG control cannot carry the SQ01 mask cue")
    workload = list(rows)
    rng = random.Random(1729)
    for _ in range(1, args.cycles):
        workload.extend(rng.sample(rows, len(rows)))
    results = []
    for load in (0.5, 0.9, 1.2):
        for burst in (False, True):
            jobs, horizon = trace(workload, load, burst=burst)
            results.append(dict(nominal_load=load, burst=burst, horizon_ms=horizon,
                                offered_service_to_horizon=sum(o.decode_ms + o.reader_ms for o in workload) / horizon,
                                policies={name: replay(jobs, horizon, name) for name in POLICIES}))
    output = dict(scope="Development replay. Synthetic captures, four serial sender queues, ideal delivery, "
                        "one nonpreemptive receiver. Deadline six median service times from capture.",
                  codec=receipt["codec"], source_sha256=hashlib.sha256(raw).hexdigest(),
                  extension_sha256=extension_hash, source_audit_sha256=source_audit_hash,
                  cycles=args.cycles, unique_measured_images=len(rows),
                  cue_bytes=receipt["cue_bytes"], cue_diagnostic=cue_check(records, rows),
                  holdout_opened=False, results=results)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

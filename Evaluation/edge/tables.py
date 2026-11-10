"""Per-image table utilities: merge JSONL partials, validate rows, build simulator jobs.

Reuses ``arrivals.build_jobs`` and the ``contracts`` dataclasses; never touches policies.
"""
import json
import math
from pathlib import Path

from Evaluation.edge.arrivals import build_jobs
from Evaluation.edge.contracts import Header, LatencyProfile, Truth

CODECS = ("v0", "v1", "jpeg")
SPLITS = ("train", "val", "test")
STS_LEN = 13
REQUIRED = ("image_id", "event", "source", "tier", "tier_probs", "relevance", "sem_key", "perc_key",
            "sts", "payload_bytes", "fidelity", "dup_group", "mission", "codec", "split")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rows(path):
    """Load a table from one JSONL file, or merge several (a list/tuple of paths)."""
    if isinstance(path, (list, tuple)):
        return merge_partial_tables(path)
    return read_jsonl(path)


def merge_partial_tables(paths):
    """Merge JSONL partials by image_id; later files fill missing/None keys, conflicts raise."""
    merged = {}
    for path in paths:
        for row in read_jsonl(path):
            if "image_id" not in row:
                raise ValueError("row without image_id in %s" % path)
            cur = merged.setdefault(row["image_id"], {})
            for k, v in row.items():
                if cur.get(k) is None:
                    cur[k] = v
                elif v is not None and cur[k] != v:
                    raise ValueError("conflict for image %r key %r: %r vs %r"
                                     % (row["image_id"], k, cur[k], v))
    return list(merged.values())


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_row(row):
    """Raise ValueError when a row violates the table schema; return the row otherwise."""
    def bad(msg):
        raise ValueError("image %r: %s" % (row.get("image_id"), msg))
    for k in REQUIRED:
        if k not in row:
            bad("missing key %s" % k)
    if not (isinstance(row["tier"], int) and 0 <= row["tier"] <= 3):
        bad("tier must be int 0..3")
    p = row["tier_probs"]
    if not (isinstance(p, (list, tuple)) and len(p) == 4 and all(_num(x) and 0 <= x <= 1 for x in p)
            and abs(sum(p) - 1.0) < 1e-3):
        bad("tier_probs must be 4 probabilities summing to 1")
    if not (_num(row["relevance"]) and -1.0 <= row["relevance"] <= 1.0):
        bad("relevance out of range")
    for k in ("sem_key", "perc_key"):
        if not (isinstance(row[k], int) and 0 <= row[k] < 2 ** 64):
            bad("%s must be a 64-bit unsigned int" % k)
    s = row["sts"]
    if not (isinstance(s, (list, tuple)) and (len(s) == 0 or (len(s) == STS_LEN and all(_num(x) for x in s)))):
        bad("sts must be empty or %d floats" % STS_LEN)
    if not (isinstance(row["payload_bytes"], int) and row["payload_bytes"] > 0):
        bad("payload_bytes must be a positive int")
    if row["fidelity"] not in (0, 1):
        bad("fidelity must be 0 or 1")
    if row["dup_group"] is not None and not isinstance(row["dup_group"], int):
        bad("dup_group must be int or null")
    if not isinstance(row["mission"], int):
        bad("mission must be int")
    if row["codec"] not in CODECS:
        bad("codec must be one of %s" % (CODECS,))
    if row["split"] not in SPLITS:
        bad("split must be one of %s" % (SPLITS,))
    return row


def load_split(rows, split):
    if split not in SPLITS:
        raise ValueError("unknown split %r" % split)
    return [r for r in rows if r["split"] == split]


def default_header_fn(row, arrival_time, capture_time):
    return Header(capture_time=capture_time, arrival_time=arrival_time,
                  payload_bytes=int(row["payload_bytes"]), mission=int(row["mission"]),
                  codec_id=CODECS.index(row["codec"]), m_index=int(row.get("m_index", 0)),
                  tier_probs=tuple(float(x) for x in row["tier_probs"]), relevance=float(row["relevance"]),
                  sem_key=int(row["sem_key"]), perc_key=int(row["perc_key"]),
                  sts=tuple(float(x) for x in row["sts"]))


def default_truth_fn(row):
    return Truth(tier=int(row["tier"]), fidelity=int(row["fidelity"]), dup_group=row["dup_group"],
                 event=row["event"] or "", source=row["source"] or "")


def rows_to_jobs(rows, times, rng, network_delay=0.0, header_fn=default_header_fn, truth_fn=default_truth_fn):
    """Jobs sorted by arrival, rows drawn without replacement."""
    return build_jobs(rows, times, header_fn, truth_fn, rng, network_delay)


def latency_from_profile_json(path):
    """LatencyProfile from JSON ``{"<batch>": [seconds...]}`` or ``{"samples": {...}}``."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return LatencyProfile(obj.get("samples", obj))

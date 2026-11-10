"""Replay saved development arrivals with LambdaMART before and after COCO masking."""

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import subprocess

import numpy as np
from PIL import Image
from Codec.Baselines.JPEG.jpeg_codec import JPEGCodec
from Evaluation.trial_scheduler_packet import CRC, pack, unpack

from Evaluation.compact_scene_transport import CompactSceneTransport
from Evaluation.no_coco_scene_transport import NoCocoSceneTransport
from Evaluation.scheduler_trial_no_coco_audit import cached_file, reconstruct
from Evaluation.semantic_learning_env import LEVELS, Environment, Episode, Message, Rule
from Scheduler.semantic_learning_policy import NumpyRanker


RUN = ("experiments/master_study/runs/launch-20261101/v9/retrieved/experiments/stream_scheduler/"
       "runs/master-study-20261101T062547Z/results/D")


def saved(revision, path):
    return json.loads(subprocess.check_output(["git", "show", f"{revision}:{RUN}/{path}"]))


def evaluate(policy, episode):
    env = Environment(episode)
    while not env.done:
        env.step(policy.select(env.snapshot()))
    result = env.metrics()
    return dict(penalty=result["J_D"], unfinished=result["offered"] - result["completed"],
                p95_capped_wait=float(np.quantile([
                    min(16, env.selected.get(m.token, episode.horizon) - m.arrival)
                    for m in episode.messages
                ], 0.95)), by_class=result["by_reference_tier"],
                order=[e["token"] for e in env.events])


def receive(message, transport, payload, capture, budget, remove_coco=False):
    legacy = transport.legacy if remove_coco else transport
    wire = reconstruct(message.features, legacy)
    packet = pack(message, transport.encode(wire), payload, capture, budget)
    received, body, received_capture, received_budget = unpack(packet, transport, message.profile_id)
    expected = message.features.copy()
    if remove_coco:
        for offset in (0, 93, 186, 279):
            expected[offset:offset + 80] = 0
    if (not np.array_equal(expected, received.features) or body != payload
            or received_capture != capture or received_budget != budget
            or any(getattr(received, name) != getattr(message, name)
                   for name in ("token", "source", "session", "sequence", "arrival"))):
        raise ValueError("received packet changed scheduler inputs or payload")
    return received, body


REDUCED_COLUMNS = np.array([i for i in range(387) if i >= 372 or i % 93 >= 80])


class ReducedRanker:
    """Predict one-unit postponement cost from retained fields and current wait."""

    def __init__(self, features, references, provenance, development):
        blocked = [row for row in provenance if row["image_id"] in development]
        groups = {row["group"] for row in blocked}
        hashes = {row["sha256"] for row in blocked}
        ids = [row["image_id"] for row in provenance if row["split"] == "train"
               and row["group"] not in groups and row["sha256"] not in hashes
               and row["image_id"] in features and row["image_id"] in references]
        if not ids:
            raise ValueError("no isolated training groups")
        x = np.stack([features[key][REDUCED_COLUMNS] for key in ids])
        budgets = np.array([(2., 4., 8., 16.)[references[key]] for key in ids])
        waits = np.array([0., 1., 2., 3., 4., 7., 8., 15., 16., 24.])
        w = waits[None, :] / budgets[:, None]
        future = (waits[None, :] + 1.) / budgets[:, None]
        target = future + np.maximum(future - 1., 0.) - w - np.maximum(w - 1., 0.)
        design = np.column_stack((np.repeat(x, len(waits), axis=0), np.tile(waits, len(x))))
        self.model = HistGradientBoostingRegressor(max_iter=40, max_leaf_nodes=15, early_stopping=False,
                                                  random_state=1729)
        with threadpool_limits(limits=2):
            self.model.fit(design, target.ravel())
        self.training_images = len(ids)

    def select(self, snapshot):
        ready = snapshot.ready
        design = np.array([np.append(m.features[REDUCED_COLUMNS], snapshot.now - m.arrival) for m in ready])
        return ready[int(np.argmax(self.model.predict(design)))].token

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision")
    parser.add_argument("output", type=Path)
    parser.add_argument("--images", required=True, type=Path)
    args = parser.parse_args()
    with np.load(io.BytesIO(cached_file(args.revision, "received_features.npz")), allow_pickle=False) as data:
        features = dict(zip(data["image_ids"], data["features"]))
    episodes = [saved(args.revision, f"baselines/fifo/episode_{i}.json") for i in range(15)]
    model = NumpyRanker(saved(args.revision, "models/lambdamart/1729/model.json"))
    profile = model.pre["profile_identity"]
    transport = NoCocoSceneTransport(CompactSceneTransport(
        1, profile["vocabulary_sha256"], profile["producer_sha256"], [True] * 85 + [False] * 8
    ))
    records_path = ("experiments/mini_dataset/runs/mini_spi_20261030_v1/"
                    "training_input/mini_training_records.json")
    records = json.loads(subprocess.check_output(["git", "show", f"{args.revision}:{records_path}"]))
    tiers_by_hash = defaultdict(set)
    for record in records:
        if record["level"] in LEVELS:
            tiers_by_hash[record["image_sha256"]].add(LEVELS.index(record["level"]))
    provenance = list(map(json.loads, cached_file(args.revision, "provenance.jsonl").splitlines()))
    reference = {}
    for row in provenance:
        tiers = tiers_by_hash.get(row["sha256"], set())
        if len(tiers) == 1:
            reference[row["image_id"]] = next(iter(tiers))
    for log in episodes:
        image_by_token = {row["token"]: row["image_id"] for row in log["metadata"]["offered_trace"]}
        for event in log["events"]:
            ident = image_by_token[event["token"]]
            tier = event["reference_tier"]
            if event["failed"] or reference.get(ident) != tier:
                raise ValueError("unexpected failed decode or conflicting saved reference")
    codec = JPEGCodec(55)
    payloads = []
    for path in sorted(args.images.glob("*.jpg")):
        with Image.open(path) as image:
            payloads.append(codec.encode(image))
    if not payloads:
        raise ValueError("at least one development JPEG source image is required")
    original_sizes, reduced_sizes = [], []
    for i, row in enumerate(features.values()):
        msg = Message(i, 0, 0, 1.0, i, row, profile_id=profile["sha256"])
        payload = payloads[i % len(payloads)]
        old, _ = receive(msg, transport.legacy, payload, 0.0, 16.0)
        new, _ = receive(msg, transport, payload, 0.0, 16.0, True)
        original_sizes.append(len(old.header_bytes) + CRC.size)
        reduced_sizes.append(len(new.header_bytes) + CRC.size)
    development = {row["image_id"] for log in episodes for row in log["metadata"]["offered_trace"]}
    ranker = ReducedRanker(features, reference, provenance, development)
    decoded_payloads = set()
    results, skipped = [], []
    for i, log in enumerate(episodes):
        meta = log["metadata"]
        if meta["input_profile"] != profile:
            raise ValueError("development log and model profile differ")
        cfg = meta["config"]
        offered = meta["offered_trace"]
        ids = tuple(row["image_id"] for row in offered)
        if any(key not in reference or key not in features for key in ids):
            skipped.append(i)
            continue
        messages, reduced_messages = [], []
        for row, key in zip(offered, ids):
            source = Message(row["token"], row["ue"], 0, row["arrival"], row["token"], features[key],
                             profile_id=profile["sha256"])
            payload = payloads[source.token % len(payloads)]
            capture = row["capture"]
            budget = (2.0, 4.0, 8.0, 16.0)[int(np.argmax(source.features[374:378]))]
            old, old_payload = receive(source, transport.legacy, payload, capture, budget)
            new, new_payload = receive(source, transport, payload, capture, budget, True)
            digest = hashlib.sha256(payload).hexdigest()
            if digest not in decoded_payloads:
                old_image, new_image = codec.decode(old_payload), codec.decode(new_payload)
                if old_image.size != new_image.size or old_image.tobytes() != new_image.tobytes():
                    raise ValueError("reconstructed pixels changed")
                decoded_payloads.add(digest)
            messages.append(old)
            reduced_messages.append(new)
        full = Episode(tuple(messages), tuple(reference[key] for key in ids), ids,
                       (cfg["service_time"],) * len(ids), (False,) * len(ids),
                       cfg["duration"] + cfg["capture_to_arrival"] + cfg["tail"], meta)
        reduced = replace(full, messages=tuple(reduced_messages))
        fifo = evaluate(Rule("fifo"), full)
        if fifo["order"] != [e["token"] for e in log["events"]] or not np.isclose(fifo["penalty"], log["metrics"]["J_D"]):
            raise ValueError("saved FIFO did not reproduce")
        spi = evaluate(Rule("spi_fifo"), full)
        if spi != evaluate(Rule("spi_fifo"), reduced):
            raise ValueError("SPI rule changed after scene-only projection")
        results.append(dict(episode=i, load=cfg["load"], fifo=fifo, spi=spi,
                            lambdamart_full=evaluate(model, full), lambdamart_no_coco=evaluate(model, reduced),
                            refit_no_coco=evaluate(ranker, reduced)))
    if not results:
        raise ValueError("no complete development traces")
    keys = ("fifo", "spi", "lambdamart_full", "lambdamart_no_coco", "refit_no_coco")
    summary = {key: {metric: float(np.mean([r[key][metric] for r in results]))
                    for metric in ("penalty", "unfinished", "p95_capped_wait")} for key in keys}
    output = dict(scope="Saved development arrivals, silver reference tiers, synthetic service. "
                  "Cached SPI and scene features; historical dynamic relevance and novelty absent in both arms.",
                  header_scope="Trial envelope including routing, timing, SPI/R/N, lengths and CRC; excludes JPEG payload and link framing.",
                  full_headers=dict(rows=len(original_sizes), original_mean=float(np.mean(original_sizes)),
                                    reduced_mean=float(np.mean(reduced_sizes)), original_p95=float(np.quantile(original_sizes, .95)),
                                    reduced_p95=float(np.quantile(reduced_sizes, .95))),
                  equal_reconstructed_jpeg_payloads=len(decoded_payloads),
                  refit_training_images=ranker.training_images,
                  revision=args.revision, model_sha256=model.artifact["sha256"], summary=summary,
                  skipped_missing_reference_or_features=skipped,
                  changed_lambdamart_orders=sum(r["lambdamart_full"]["order"] != r["lambdamart_no_coco"]["order"]
                                               for r in results),
                  episodes=[{**r, **{key: {k: v for k, v in r[key].items() if k != "order"} for key in keys}}
                            for r in results])
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

"""Replay saved development arrivals with LambdaMART before and after COCO masking."""

import argparse
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from Evaluation.compact_scene_transport import CompactSceneTransport
from Evaluation.no_coco_scene_transport import NoCocoSceneTransport
from Evaluation.scheduler_trial_no_coco_audit import cached_file, reconstruct
from Evaluation.semantic_learning_env import Environment, Episode, Message, Rule
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


def reduced_message(message, transport):
    wire = reconstruct(message.features, transport.legacy)
    packet = transport.encode(wire)
    decoded = transport.decode(packet)
    features = message.features.copy()
    for offset in (0, 93, 186, 279):
        features[offset:offset + 80] = 0
    if reconstruct(features, transport.legacy) != decoded:
        raise ValueError("reduced model vector does not match decoded scene")
    return replace(message, features=features, header_bytes=packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(io.BytesIO(cached_file(args.revision, "received_features.npz")), allow_pickle=False) as data:
        features = dict(zip(data["image_ids"], data["features"]))
    episodes = [saved(args.revision, f"baselines/fifo/episode_{i}.json") for i in range(15)]
    model = NumpyRanker(saved(args.revision, "models/lambdamart/1729/model.json"))
    profile = model.pre["profile_identity"]
    transport = NoCocoSceneTransport(CompactSceneTransport(
        1, profile["vocabulary_sha256"], profile["producer_sha256"], [True] * 85 + [False] * 8
    ))
    reference = {}
    for log in episodes:
        image_by_token = {row["token"]: row["image_id"] for row in log["metadata"]["offered_trace"]}
        for event in log["events"]:
            ident = image_by_token[event["token"]]
            tier = event["reference_tier"]
            if event["failed"] or reference.setdefault(ident, tier) != tier:
                raise ValueError("unexpected failed decode or conflicting saved reference")
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
        messages = tuple(Message(row["token"], row["ue"], 0, row["arrival"], row["token"], features[key],
                                 profile_id=profile["sha256"]) for row, key in zip(offered, ids))
        full = Episode(messages, tuple(reference[key] for key in ids), ids,
                       (cfg["service_time"],) * len(ids), (False,) * len(ids),
                       cfg["duration"] + cfg["capture_to_arrival"] + cfg["tail"], meta)
        reduced = replace(full, messages=tuple(reduced_message(m, transport) for m in messages))
        fifo = evaluate(Rule("fifo"), full)
        if fifo["order"] != [e["token"] for e in log["events"]] or not np.isclose(fifo["penalty"], log["metrics"]["J_D"]):
            raise ValueError("saved FIFO did not reproduce")
        spi = evaluate(Rule("spi_fifo"), full)
        if spi != evaluate(Rule("spi_fifo"), reduced):
            raise ValueError("SPI rule changed after scene-only projection")
        results.append(dict(episode=i, load=cfg["load"], fifo=fifo, spi=spi,
                            lambdamart_full=evaluate(model, full), lambdamart_no_coco=evaluate(model, reduced)))
    if not results:
        raise ValueError("no complete development traces")
    keys = ("fifo", "spi", "lambdamart_full", "lambdamart_no_coco")
    summary = {key: {metric: float(np.mean([r[key][metric] for r in results]))
                    for metric in ("penalty", "unfinished", "p95_capped_wait")} for key in keys}
    output = dict(scope="Saved development arrivals, silver reference tiers, synthetic service. "
                  "Cached SPI and scene features; historical dynamic relevance and novelty absent in both arms.",
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

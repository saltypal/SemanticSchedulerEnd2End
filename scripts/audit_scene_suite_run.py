"""Read-only run audit and bounded validation replay; never trains or opens pilot_eval."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def mean(values):
    return float(np.mean(values)) if values else None


def audit_archive(archive, output):
    from Scheduler.training.scene_corpus import default_registry
    from Scheduler.training.scene_study import restore_context
    from Scheduler.scene_policy import context_matrix

    with zipfile.ZipFile(archive) as z:
        read = lambda name: json.loads(z.read(name))
        checks = read("artifact_checksums.json")
        bad = [p for p, h in checks.items() if hashlib.sha256(z.read(p)).hexdigest() != h]
        if bad:
            raise ValueError(f"Artifact hash mismatch: {bad}")
        source = read("source_manifest.json")
        inspected = ["Scheduler/training/scene_corpus.py", "Scheduler/training/scene_study.py",
                     "Scheduler/training/scene_learners.py", "Evaluation/scene_environment.py",
                     "Evaluation/scene_queue.py", "Scheduler/scene_policy.py"]
        matched = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == source["files"][p]
                   for p in inspected}
        if not all(matched.values()):
            raise ValueError("Replay requires the audited implementation to match submitted source")
        summary = read("study/summary.json")
        stats, curves, details = {}, {}, {}
        for row in summary:
            name = row["algorithm"]
            if row["status"] != "complete":
                continue
            rows = read(f"study/algorithms/{name}/validation.json")
            details[name] = rows
            cells = defaultdict(list)
            for r in rows:
                cells[(r["load"], r["pattern"])].append(r["J_time"])
            stats[name] = {
                "J": mean([r["J_time"] for r in rows]),
                "J_equal_nine_cells": mean([mean(v) for v in cells.values()]),
                "by_load": {load: {"n": sum(r["load"] == load for r in rows),
                    "J": mean([r["J_time"] for r in rows if r["load"] == load]),
                    "final_coverage": mean([r["coverage_at_horizon"] for r in rows if r["load"] == load])}
                    for load in ("light", "congested", "burst")},
                "by_pattern": {p: mean([r["J_time"] for r in rows if r["pattern"] == p])
                    for p in ("no_repeat", "local_repeat", "cross_repeat")},
                "reward_integral_max_error": max(abs(r["J_time"] - r["reward_sum"]) for r in rows),
                "mean_header_bytes": mean([r["header_bytes_offered"] / 48 for r in rows]),
                "fallbacks": sum(r["fallback_count"] for r in rows),
            }
            first_ten = [r for r in rows if len(r["curve"]) >= 10 and r["offered_weight"] > 0]
            stats[name]["completion_budget_10"] = {
                "defined_traces": len(first_ten),
                "mean_distinct_useful_in_first_10": mean([r["curve"][9]["covered_weight"] for r in first_ten]),
                "mean_coverage_at_10": mean([r["curve"][9]["covered_weight"] / r["offered_weight"] for r in first_ten])}
            curve = read(f"study/algorithms/{name}/learning_curve.json")
            curves[name] = curve
            scored = [r for r in curve if r.get("validation_J", r.get("validation_score")) is not None]
            if scored:
                get_score = lambda r: r.get("validation_J", r.get("validation_score"))
                stats[name]["learning"] = {"first": scored[0], "best": max(scored, key=get_score),
                                           "last": scored[-1], "records": len(curve)}
            if name != "damage_probability":
                base = details.get("damage_probability")
                if base:
                    delta = [a["J_time"] - b["J_time"] for a, b in zip(rows, base)]
                    stats[name]["versus_damage_probability"] = {
                        "mean_delta": mean(delta), "wins": sum(d > 1e-12 for d in delta),
                        "ties": sum(abs(d) <= 1e-12 for d in delta), "losses": sum(d < -1e-12 for d in delta),
                        "min_delta": min(delta), "max_delta": max(delta),
                        "note": "Matched trace descriptions; not independent-group uncertainty."}

        examples = [json.loads(line) for line in z.read("study/training_decisions.jsonl").decode().splitlines()]
        training = Counter()
        available = []; raw = []; labels = []
        pair_total = Counter(); pair_kept = Counter(); aliased = defaultdict(set)
        d = len(default_registry().fields)
        medians = [0.] * d
        for e in examples:
            c = restore_context(e["context"])
            gains = e["gains"]
            training["snapshots"] += 1
            training["candidate_rows"] += len(gains)
            training["all_zero_decisions"] += int(max(gains) == 0)
            training["singleton_decisions"] += int(len(gains) == 1)
            training["nontrivial_decisions"] += int(len(set(gains)) > 1)
            pairs = [(i, j) for i in range(len(gains)) for j in range(len(gains)) if gains[i] > gains[j]]
            training["pair_truncated_decisions"] += int(len(pairs) > 32)
            if len(pairs) > 32:
                pair_total.update(i for i, _ in pairs)
                pair_kept.update(i for i, _ in pairs[:32])
            matrix = context_matrix(c, medians)
            for i, candidate in enumerate(c.ready_candidates):
                training[f"gain_{gains[i]}"] += 1
                available.append(candidate.available); raw.append(candidate.values); labels.append(gains[i])
                aliased[hashlib.sha256(matrix[i].tobytes()).hexdigest()].add(gains[i])
        training["episodes"] = len({e["episode"] for e in examples})
        training["singleton_and_zero_intersection"] = sum(
            len(e["gains"]) == 1 and max(e["gains"]) == 0 for e in examples)
        training["all_equal_gain_decisions"] = sum(len(set(e["gains"])) == 1 for e in examples)
        a = np.asarray(available, bool); x = np.asarray(raw)
        field_stats = []
        for i, name in enumerate(default_registry().fields):
            vals = x[a[:, i], i]
            field_stats.append({"field": name, "availability": float(a[:, i].mean()),
                "distinct_available_values": int(len(np.unique(vals))),
                "min": float(vals.min()) if len(vals) else None,
                "max": float(vals.max()) if len(vals) else None})
        training["never_available_fields"] = sum(r["availability"] == 0 for r in field_stats)
        training["constant_or_unavailable_value_fields"] = sum(r["distinct_available_values"] <= 1 for r in field_stats)
        training["exact_context_feature_conflict_keys"] = sum(len(v) > 1 for v in aliased.values())
        training["exact_context_feature_keys"] = len(aliased)
        training["ranknet_retained_by_positive_position"] = {
            str(i): {"available_pairs": pair_total[i], "retained_pairs": pair_kept[i]} for i in sorted(pair_total)}
        result = {"archive": str(archive.resolve()), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                  "verified_artifacts": len(checks), "source_matches": matched,
                  "config": read("study/config.json"), "status": read("status.json"),
                  "environment": read("environment.json"), "summary": summary,
                  "statistics": stats, "training": dict(training), "fields": field_stats}
        save(output / "archive_audit.json", result)
        save(output / "completion_budget_metrics.json", {n: v["completion_budget_10"] for n, v in stats.items()})
        plot_archive(output, stats, curves)
        print("Archive verified:", len(checks), "files; training snapshots:", len(examples), flush=True)
        return result


def producer_validation(data, output):
    """Analyze cached producer predictions only; no inference or pilot-evaluation labels."""
    rows = [r for r in json.loads((data / "records.json").read_text()) if r["role"] == "validation"]
    features = {r["image_id"]: r["evidence"] for r in map(json.loads, (data / "features.jsonl").read_text().splitlines())}
    result = {}
    for source in ("all", "train", "test"):
        subset = [r for r in rows if source == "all" or r["source_split"] == source]
        values = [(float(features[r["image_id"]]["probabilities"]["building.affected_or_greater"]),
                   int(r["damage_positive"])) for r in subset]
        positives = [p for p, y in values if y]; negatives = [p for p, y in values if not y]
        auc = sum((p > q) + .5 * (p == q) for p in positives for q in negatives) / (len(positives) * len(negatives)) if positives and negatives else None
        result[source] = {"n": len(values), "positive": len(positives), "AUROC": auc,
            "TP": sum(p > .5 and y == 1 for p, y in values),
            "FP": sum(p > .5 and y == 0 for p, y in values),
            "FN": sum(p <= .5 and y == 1 for p, y in values),
            "TN": sum(p <= .5 and y == 0 for p, y in values),
            "brier": mean([(p - y) ** 2 for p, y in values])}
    result["scope"] = ("Cached validation producer only, not scheduling accuracy; source-train images "
                       "exposed to frozen image classifier; tiny strata, no significance claim.")
    save(output / "producer_validation.json", result)


def plot_archive(output, stats, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered = sorted(stats, key=lambda n: stats[n]["J"])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(ordered, [stats[n]["J"] for n in ordered])
    ax.set(xlabel="Validation time-integrated positive-observation coverage J", xlim=(0, 1),
           title="One-seed pilot: 32 matched traces, synthetic service time")
    fig.tight_layout(); fig.savefig(output / "validation_scores.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ("masked_ppo", "masked_double_dqn", "td_per"):
        rows = [r for r in curves[name] if r.get("validation_J") is not None]
        ax.plot([r["step"] for r in rows], [r["validation_J"] for r in rows], marker=".", label=name)
    ax.axhline(stats["damage_probability"]["J"], color="black", linestyle="--", label="damage score baseline")
    ax.set(xlabel="Training action decisions", ylabel="Validation J", title="Later RL checkpoints degrade in this run")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "rl_learning_curves.png", dpi=180); plt.close(fig)


def replay(archive, data, upstream, output, count):
    from Scheduler.training.scene_corpus import CausalCorpus
    from Evaluation.scene_environment import SceneEnvironment
    from Scheduler.scene_policy import ScenePolicy, tie_key
    corpus = CausalCorpus(data, upstream)
    damage = corpus.registry.fields.index("claim/building.affected_or_greater/score")
    # Exact runtime evidence signature, excluding offered-stream KB and time fields.
    signature_fields = [i for i, f in enumerate(corpus.registry.fields)
                        if f.startswith(("claim/", "detection/"))]
    def signature(c):
        return tuple((c.available[i], c.values[i] if c.available[i] else None) for i in signature_fields)
    models = {}
    with zipfile.ZipFile(archive) as z:
        for name in ("lambdamart", "masked_ppo", "masked_double_dqn", "td_per"):
            models[name] = json.loads(z.read(f"study/algorithms/{name}/model.json"))
    names = ["damage_probability", "damage_header_repeat_guard", *models, "privileged_greedy"]
    outcomes = defaultdict(list); own_decisions = defaultdict(Counter)
    for i in range(count):
        begun = time.perf_counter()
        pattern = ("no_repeat", "local_repeat", "cross_repeat")[i % 3]
        load = ("light", "congested", "burst")[(i // 3) % 3]
        base = corpus.make("validation", 41000 + i, pattern=pattern, load=load)
        for name in names:
            env = SceneEnvironment(base.episode, base.registry, base.mission, base.specifications)
            policy = ScenePolicy(models[name]) if name in models else ScenePolicy(kind="single", field=damage)
            decisions = own_decisions[name]
            while not env.finished:
                ctx = env.observation(); gains = env.marginal_gains(); rows = ctx.ready_candidates
                decisions["decisions"] += 1; decisions["singletons"] += int(len(rows) == 1)
                if name == "privileged_greedy":
                    selected = min(range(len(rows)), key=lambda j: (-gains[j], tie_key(rows[j])))
                elif name == "damage_header_repeat_guard":
                    seen = {signature(c) for c in ctx.successful_header_history}
                    selected = min(range(len(rows)), key=lambda j: (
                        signature(rows[j]) in seen,
                        -rows[j].values[damage] if rows[j].available[damage] else float("inf"), tie_key(rows[j])))
                else:
                    decision = policy.select(ctx)
                    selected = next(j for j, c in enumerate(rows) if c.message_id == decision.message_id)
                if len(set(gains)) > 1:
                    decisions["informative_decisions"] += 1
                    decisions["best_gain_choices"] += int(gains[selected] == max(gains))
                    decisions["immediate_gain_regret"] += max(gains) - gains[selected]
                env.step(rows[selected].message_id)
            metrics = env.metrics()
            # Arrival-only upper bound ignores queue contention and is not an achievable oracle.
            first_arrival = {f: min(m.arrival_time for m in base.episode.messages
                              if f in base.episode.facts[m.message_id]) for f in base.episode.weights}
            release_bound = sum(w * max(base.episode.horizon - first_arrival[f] - 1, 0)
                                / base.episode.horizon for f, w in base.episode.weights.items()) / env.total
            outcomes[name].append({"trace": i, "load": load, "pattern": pattern,
                "J": metrics["J_time"], "final_coverage": metrics["coverage_at_horizon"],
                "repeat_fraction": metrics["redundant_weight_fraction"],
                "arrival_only_upper_bound": release_bound,
                "mean_delivered_positive_delay": mean(metrics["exposure_delays"]),
                "implied_mean_completion_time": base.episode.horizon * (1 - metrics["J_time"]),
                "horizon": base.episode.horizon})
        print(f"Validation replay {i+1}/{count}: {load}/{pattern}; {time.perf_counter()-begun:.2f}s", flush=True)
    for name, rows in outcomes.items():
        for r, oracle in zip(rows, outcomes["privileged_greedy"]):
            r["J_regret_to_privileged_greedy"] = oracle["J"] - r["J"]
    save(output / "validation_replay.json", {"count": count, "role": "validation_only",
        "diagnostic_is_new_development_experiment": True,
        "history_limit": 32, "outcomes": dict(outcomes), "visited_state_choice_counts": dict(own_decisions),
        "warning": "Decision counts describe each policy's own states. They are not a shared-state choice benchmark."})
    print("Replay means:", {n: mean([r["J"] for r in rs]) for n, rs in outcomes.items()}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--upstream", type=Path)
    p.add_argument("--replay-traces", type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    print("START read-only audit:", args.archive, "output:", args.output, flush=True)
    audit_archive(args.archive, args.output)
    if args.data:
        producer_validation(args.data, args.output)
    if args.replay_traces:
        if not args.data or not args.upstream or not 1 <= args.replay_traces <= 32:
            raise ValueError("Replay requires frozen data/upstream and 1..32 validation traces")
        replay(args.archive, args.data, args.upstream, args.output, args.replay_traces)
    print("END audit", flush=True)


if __name__ == "__main__":
    main()

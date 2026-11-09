"""Pooled eight-value scene descriptor and RankNet training path (Test 1).

Frozen pre-mask grid -> global mean/max pooling -> small learned projection
into eight bounded scene values -> shared linear RankNet scorer trained with
pairwise logistic loss on independently reviewed same-tier scene pairs.

Design boundaries enforced by this module:
- Mean pooling carries broad scene context, max pooling gives localized
  evidence a route into the descriptor.
- Training scores fake-uint8 quantized values; evaluation uses exact uint8.
- A one-value projection is the size/complexity ablation.
- No wait, service time, identity or history value ever enters the inputs or
  the targets: inputs are pooled grid values only, targets are ordered pairs
  from reviewed grades only.
- Labels are never fabricated. Only a manifest declaring
  ``review_kind = "independent_scene_pair_review"`` with per-pair reviewers is
  accepted; uncertain and tie labels are excluded and counted, and native
  road-label records are rejected outright.
- The active Qwen/Nemotron decision adds a second, explicitly distinct
  manifest kind, ``review_kind = "vlm_scene_pair_review"`` (see
  ``scene_vlm_review``). It must carry pinned model revisions, a prompt
  version, per-image split plus event/flight/near-duplicate group IDs, and
  forward/reverse votes from both reviewers for both image orders. VLM
  labels are silver consensus kept apart from human review; only stable
  jointly supported pairs train. This module also exposes non-raising
  reporting helpers: stable swapped-order agreement metrics (90% order
  consistency, 50% stable strict coverage, 80% cross-model agreement with
  overlap size) and a group-disjoint cohort report, so a shortfall can be
  recorded instead of only raised. Nothing here invents a label, grade,
  group, split or judgment.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from Scheduler.training.scene_vlm_review import (
    ANSWERS,
    REVIEW_KIND as VLM_REVIEW_KIND,
    SPLITS,
    stable_choice,
)

REVIEW_KIND = "independent_scene_pair_review"
# Image-record group dimensions frozen across splits (scene_vlm_review names).
GROUP_DIMENSIONS = ("event", "flight", "near_duplicate")
# Qualification gates from the active Qwen/Nemotron review decision; these
# mirror the thresholds scene_vlm_review enforces inside load_reviewed_pairs.
ORDER_CONSISTENCY_MIN = 0.90
STABLE_STRICT_COVERAGE_MIN = 0.50
CROSS_MODEL_AGREEMENT_MIN = 0.80
GRADES = (0, 1, 2)
PAIR_STATUS = ("strict", "uncertain", "tie")
# Records that carry native detector labels instead of independent review.
NATIVE_LABEL_KEYS = ("positive_labels", "native_labels_json", "roads_damage")


class ReviewedPairUnavailable(RuntimeError):
    """The independently reviewed same-tier pair data does not exist yet."""


def pool_grid(grid):
    """Global mean and max pooling over the spatial axes -> (2C,) tensor."""
    tensor = torch.as_tensor(grid, dtype=torch.float32)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("pool_grid expects a single image grid")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError("grid must be (C,H,W) or (1,C,H,W), got %r"
                         % (tuple(tensor.shape),))
    spatial = tensor.reshape(tensor.shape[0], -1)
    return torch.cat((spatial.mean(dim=1), spatial.max(dim=1).values), dim=0)


class PooledSceneProjection(nn.Module):
    """Small learned projection from pooled features to bounded scene values."""

    def __init__(self, channels, values=8):
        super().__init__()
        if channels < 1 or values < 1:
            raise ValueError("channels and values must be positive")
        self.channels = int(channels)
        self.values = int(values)
        self.projection = nn.Linear(2 * self.channels, self.values)

    def forward(self, pooled):
        return torch.sigmoid(self.projection(pooled))


class RankNetScorer(nn.Module):
    """Shared linear score over the compact scene descriptor."""

    def __init__(self, values=8):
        super().__init__()
        self.values = int(values)
        self.scorer = nn.Linear(self.values, 1)

    def forward(self, scene):
        return self.scorer(scene).squeeze(-1)


def fake_uint8(values):
    """Fake uint8 quantization with straight-through gradients for training."""
    quantized = torch.round(values.detach() * 255.0) / 255.0
    return values + (quantized - values).detach()


def exact_uint8(values):
    """Exact uint8 quantization for evaluation."""
    array = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def pairwise_logistic_loss(preferred, other):
    """RankNet pairwise logistic loss on ordered score differences."""
    return F.softplus(-(preferred - other)).mean()


def _reject_native_labels(record, where):
    present = [key for key in NATIVE_LABEL_KEYS if key in record]
    if present:
        raise ValueError(
            "%s carries native detector labels (%s), not independent scene review"
            % (where, ", ".join(present)))


def check_group_disjointness(images, pairs=None):
    """Report group/split crossings without raising (cohort freeze check).

    ``images`` maps image id -> record with a non-empty ``split`` and the
    event/flight/near-duplicate group fields of the VLM review format.
    An unresolved dimension uses one explicit shared sentinel (for example
    ``"unresolved"``) instead of being omitted, so it can never drift
    across splits unnoticed. ``pairs`` optionally adds: both images must
    have a record, share one split and share one SPI tier. Returns a report
    with per-split and per-dimension counts, ``violations`` and
    ``disjoint``; nothing is raised, so cohort assignments merged from
    several manifests can be inspected before fitting.
    ``load_reviewed_pairs`` performs the same checks and raises on a failed
    manifest.
    """
    if not isinstance(images, dict):
        raise ValueError("images must map image id -> group/split record")
    required = ("split",) + GROUP_DIMENSIONS
    violations = []
    splits = {}
    group_splits = {dimension: {} for dimension in GROUP_DIMENSIONS}
    for image_id, record in sorted(images.items(),
                                   key=lambda item: str(item[0])):
        if not isinstance(record, dict):
            violations.append({
                "type": "malformed_image_record", "image": image_id,
                "detail": "image %r record is not an object" % (image_id,)})
            continue
        missing = [field for field in required
                   if not isinstance(record.get(field), str)
                   or not record[field].strip()]
        if missing:
            violations.append({
                "type": "malformed_image_record", "image": image_id,
                "detail": "image %r missing group/split fields: %s"
                          % (image_id, ", ".join(missing))})
            continue
        split = record["split"]
        if split not in SPLITS:
            violations.append({
                "type": "malformed_image_record", "image": image_id,
                "detail": "image %r has unknown review split %r"
                          % (image_id, split)})
            continue
        splits[split] = splits.get(split, 0) + 1
        for dimension in GROUP_DIMENSIONS:
            group_splits[dimension].setdefault(
                record[dimension], set()).add(split)
    for dimension in GROUP_DIMENSIONS:
        for group, seen in sorted(group_splits[dimension].items()):
            if len(seen) > 1:
                violations.append({
                    "type": "group_crosses_split", "dimension": dimension,
                    "group": group, "splits": sorted(seen),
                    "detail": "%s %r appears in splits %s"
                              % (dimension, group, ", ".join(sorted(seen)))})
    if pairs is not None:
        if not isinstance(pairs, list):
            raise ValueError("pairs must be a list of pair objects")
        for index, pair in enumerate(pairs):
            where = "pair %d" % index
            if (not isinstance(pair, dict)
                    or not isinstance(pair.get("a"), str)
                    or not isinstance(pair.get("b"), str)):
                violations.append({
                    "type": "malformed_pair", "pair": index,
                    "detail": "%s lacks string image ids a/b" % where})
                continue
            record_a = images.get(pair["a"])
            record_b = images.get(pair["b"])
            missing = [image_id for image_id, record in
                       ((pair["a"], record_a), (pair["b"], record_b))
                       if not isinstance(record, dict)]
            for image_id in missing:
                violations.append({
                    "type": "pair_missing_image",
                    "pair": [pair["a"], pair["b"]], "image": image_id,
                    "detail": "%s references image %r without a group/split "
                              "record" % (where, image_id)})
            if missing:
                continue
            split_a = record_a.get("split")
            split_b = record_b.get("split")
            if (isinstance(split_a, str) and isinstance(split_b, str)
                    and split_a and split_b and split_a != split_b):
                violations.append({
                    "type": "pair_crosses_split",
                    "pair": [pair["a"], pair["b"]],
                    "splits": sorted({split_a, split_b}),
                    "detail": "%s spans splits %s"
                              % (where, ", ".join(sorted({split_a, split_b})))})
            tier_a = record_a.get("spi_tier")
            tier_b = record_b.get("spi_tier")
            if type(tier_a) is not int or type(tier_b) is not int:
                violations.append({
                    "type": "pair_tier_missing",
                    "pair": [pair["a"], pair["b"]],
                    "detail": "%s lacks an integer spi_tier on both images"
                              % where})
            elif tier_a != tier_b:
                violations.append({
                    "type": "pair_tier_mismatch",
                    "pair": [pair["a"], pair["b"]],
                    "detail": "%s has different SPI tiers %r and %r"
                              % (where, tier_a, tier_b)})
    return {"images": len(images),
            "splits": dict(sorted(splits.items())),
            "dimensions": {dimension: len(group_splits[dimension])
                           for dimension in GROUP_DIMENSIONS},
            "violations": violations, "disjoint": not violations}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _calibration_decisions(manifest):
    """Map calibration-pair votes to per-model swapped-order decisions.

    Returns ``(records, models, pairs_by_split, unresolved)``. One record
    holds every calibration pair with, for each pinned reviewer, whether
    both answers were valid, whether the mapped verdicts agreed across the
    two image orders, and the stable strict choice when they did. Malformed
    or missing answers count as invalid instead of raising, so they are
    logged as a shortfall rather than fabricated; pairs whose image records
    are absent are skipped and counted as unresolved.
    """
    if not isinstance(manifest, dict):
        raise ValueError("VLM review manifest must be a JSON object")
    if manifest.get("review_kind") != VLM_REVIEW_KIND:
        raise ValueError("swapped-order metrics require review_kind=%r"
                         % VLM_REVIEW_KIND)
    model_revisions = manifest.get("model_revisions")
    if (not isinstance(model_revisions, dict) or not model_revisions
            or not all(isinstance(value, str) and value.strip()
                       for value in model_revisions.values())):
        raise ValueError(
            "VLM review requires a pinned model revision for every reviewer")
    images = manifest.get("images")
    pairs = manifest.get("pairs")
    if not isinstance(images, dict):
        raise ValueError("VLM review requires an images mapping")
    if not isinstance(pairs, list):
        raise ValueError("VLM review requires a pairs list")
    models = sorted(model_revisions)
    records = []
    pairs_by_split = {}
    unresolved = 0
    for index, pair in enumerate(pairs):
        if (not isinstance(pair, dict)
                or not isinstance(pair.get("a"), str)
                or not isinstance(pair.get("b"), str)):
            raise ValueError("pair %d lacks string image ids a/b" % index)
        a, b = pair["a"], pair["b"]
        record_a = images.get(a)
        record_b = images.get(b)
        if not isinstance(record_a, dict) or not isinstance(record_b, dict):
            unresolved += 1
            continue
        split_a = record_a.get("split")
        split_b = record_b.get("split")
        if split_a != split_b:
            raise ValueError("pair %d (%s, %s) crosses review splits"
                             % (index, a, b))
        pairs_by_split[str(split_a)] = pairs_by_split.get(str(split_a), 0) + 1
        if split_a != "calibration":
            continue
        votes = pair.get("votes")
        decisions = {}
        for model in models:
            vote = votes.get(model) if isinstance(votes, dict) else None
            forward = vote.get("forward") if isinstance(vote, dict) else None
            reverse = vote.get("reverse") if isinstance(vote, dict) else None
            valid = (isinstance(forward, str) and forward in ANSWERS
                     and isinstance(reverse, str) and reverse in ANSWERS)
            choice = stable_choice(a, b, forward, reverse) if valid else None
            decisions[model] = {
                "valid": valid,
                "consistent": valid and choice is not None,
                "stable_strict": choice in (a, b),
                "choice": choice if choice in (a, b) else None,
            }
        records.append({"a": a, "b": b, "models": decisions})
    return records, models, pairs_by_split, unresolved


def swapped_order_metrics(manifest):
    """Report stable swapped-order agreement metrics without raising.

    Per pinned reviewer over ALL calibration pairs:

    - ``order_consistency``: mapped verdicts agree across both image
      orders, among pairs with two valid responses (the 90% gate).
    - ``stable_strict_coverage``: stable strict choices among ALL
      calibration pairs; abstentions, malformed answers, ties and
      disagreements stay in the denominator (the 50% gate).

    Jointly: the strict overlap between reviewers, its size, and their
    agreement on that overlap (the 80% gate). Threshold failures come back
    as values, never as exceptions, so a shortfall can be recorded before
    advancing; only structural manifest errors raise. See
    ``qualification_report`` for the pass/fail gates.
    """
    records, models, pairs_by_split, unresolved = _calibration_decisions(
        manifest)
    total = len(records)
    model_metrics = {}
    for model in models:
        two_valid = [row for row in records if row["models"][model]["valid"]]
        consistent = [row for row in two_valid
                      if row["models"][model]["consistent"]]
        stable_strict = [row for row in records
                         if row["models"][model]["stable_strict"]]
        model_metrics[model] = {
            "pairs_two_valid_responses": len(two_valid),
            "order_consistent_pairs": len(consistent),
            "order_consistency": _ratio(len(consistent), len(two_valid)),
            "stable_strict_pairs": len(stable_strict),
            "stable_strict_coverage": _ratio(len(stable_strict), total),
        }
    overlap = [row for row in records
               if all(row["models"][model]["stable_strict"]
                      for model in models)]
    agreed = [row for row in overlap
              if len({row["models"][model]["choice"]
                      for model in models}) == 1]
    disagreements = [
        {"a": row["a"], "b": row["b"],
         "choices": {model: row["models"][model]["choice"]
                     for model in models}}
        for row in overlap if row not in agreed][:10]
    return {
        "review_kind": VLM_REVIEW_KIND,
        "split": "calibration",
        "calibration_pairs": total,
        "pairs_by_split": dict(sorted(pairs_by_split.items())),
        "pairs_unresolved": unresolved,
        "models": model_metrics,
        "joint": {
            "overlap_pairs": len(overlap),
            "overlap_fraction": _ratio(len(overlap), total),
            "agreed_pairs": len(agreed),
            "agreement": _ratio(len(agreed), len(overlap)),
            "disagreement_pairs": len(overlap) - len(agreed),
            "disagreement_examples": disagreements,
        },
        "thresholds": {"order_consistency": ORDER_CONSISTENCY_MIN,
                       "stable_strict_coverage": STABLE_STRICT_COVERAGE_MIN,
                       "cross_model_agreement": CROSS_MODEL_AGREEMENT_MIN},
    }


def qualification_report(metrics):
    """Apply the 90/50/80 gates to swapped_order_metrics without raising.

    Returns one pass/fail gate per reviewer plus the cross-model agreement
    gate, the overlap size, and a single ``qualified`` verdict. A missing
    metric (empty denominator) never passes.
    """
    if (not isinstance(metrics, dict) or "models" not in metrics
            or "joint" not in metrics or "thresholds" not in metrics):
        raise ValueError("qualification_report needs swapped_order_metrics "
                         "output")
    thresholds = metrics["thresholds"]

    def gate(value, threshold):
        return {"value": value, "threshold": threshold,
                "pass": value is not None and value >= threshold}

    models = {model: {
        "order_consistency": gate(
            values["order_consistency"], thresholds["order_consistency"]),
        "stable_strict_coverage": gate(
            values["stable_strict_coverage"],
            thresholds["stable_strict_coverage"]),
    } for model, values in metrics["models"].items()}
    agreement = gate(metrics["joint"]["agreement"],
                     thresholds["cross_model_agreement"])
    report = {
        "thresholds": thresholds,
        "models": models,
        "cross_model_agreement": agreement,
        "overlap_pairs": metrics["joint"]["overlap_pairs"],
        "qualified": agreement["pass"] and all(
            gate_result["pass"]
            for per_model in models.values()
            for gate_result in per_model.values()),
    }
    return report


def load_reviewed_pairs(path):
    """Load and strictly validate an independent same-tier scene-pair review."""
    import json
    path = Path(path)
    if not path.is_file():
        raise ReviewedPairUnavailable(
            "independently reviewed scene pairs not found at %s; the Test 1 "
            "acceptance gate stays blocked until real review data exists" % path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("review manifest must be a JSON object")
    _reject_native_labels(manifest, "review manifest")
    from Scheduler.training.scene_vlm_review import REVIEW_KIND as VLM_REVIEW_KIND, training_consensus

    if manifest.get("review_kind") == VLM_REVIEW_KIND:
        pairs, calibration = training_consensus(manifest)
        return {
            "review_kind": VLM_REVIEW_KIND,
            "strict_pairs": pairs,
            "excluded": {"non_training_or_without_consensus": len(manifest["pairs"]) - len(pairs)},
            "reviewers": sorted(manifest["model_revisions"]),
            "calibration": calibration,
            "model_revisions": manifest["model_revisions"],
            "prompt_version": manifest["prompt_version"],
            "path": str(path),
        }
    if manifest.get("review_kind") != REVIEW_KIND:
        raise ValueError(
            "manifest does not declare review_kind=%r; historical labels, tags "
            "and rubric drafts are not scene review" % REVIEW_KIND)
    if tuple(manifest.get("grades", ())) != GRADES:
        raise ValueError("review grades must be exactly 0, 1, 2")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("review manifest contains no pairs")
    strict, excluded = [], {"uncertain": 0, "tie": 0, "cross_tier": 0, "equal_grade": 0}
    reviewers = set()
    for index, pair in enumerate(pairs):
        where = "pair %d" % index
        _reject_native_labels(pair, where)
        missing = [key for key in ("a", "b", "grade_a", "grade_b", "status",
                                   "reviewer", "same_tier")
                   if key not in pair]
        if missing:
            raise ValueError("%s missing review fields: %s"
                             % (where, ", ".join(missing)))
        status = pair["status"]
        if status not in PAIR_STATUS:
            raise ValueError("%s has invalid status %r" % (where, status))
        for key in ("grade_a", "grade_b"):
            if pair[key] not in GRADES:
                raise ValueError("%s has grade outside 0/1/2" % where)
        if not str(pair["reviewer"]).strip():
            raise ValueError("%s has no named reviewer" % where)
        if pair["a"] == pair["b"]:
            raise ValueError("%s compares an image with itself" % where)
        if status != "strict":
            excluded[status] += 1
            continue
        if not pair["same_tier"]:
            excluded["cross_tier"] += 1
            continue
        if pair["grade_a"] == pair["grade_b"]:
            excluded["equal_grade"] += 1
            continue
        reviewers.add(str(pair["reviewer"]))
        strict.append(pair)
    if not strict:
        raise ValueError("review manifest has no strict same-tier ordered pairs")
    return {
        "review_kind": REVIEW_KIND,
        "strict_pairs": strict,
        "excluded": excluded,
        "reviewers": sorted(reviewers),
        "path": str(path),
    }


def ordered_pairs(pairs):
    """Split strict pairs into (preferred ids, other ids, image ids)."""
    preferred, other, image_ids = [], [], []
    for pair in pairs:
        for key in ("a", "b"):
            if pair[key] not in image_ids:
                image_ids.append(pair[key])
        if pair["grade_a"] > pair["grade_b"]:
            preferred.append(pair["a"])
            other.append(pair["b"])
        else:
            preferred.append(pair["b"])
            other.append(pair["a"])
    return preferred, other, image_ids


def load_grids(directory):
    """Load cached pre-mask grids written as ``<image_id>.npy`` arrays."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ReviewedPairUnavailable(
            "pre-mask feature cache not found at %s; run the parity smoke with "
            "--grids-out first" % directory)
    grids = {}
    for path in sorted(directory.glob("*.npy")):
        grids[path.stem] = np.load(path)
    if not grids:
        raise ReviewedPairUnavailable(
            "no pre-mask grids cached under %s" % directory)
    return grids


def train(grids, manifest_path, *, values=8, steps=200, lr=0.05, seed=0):
    """Train the pooled projection and shared RankNet scorer on reviewed pairs."""
    manifest = load_reviewed_pairs(manifest_path)
    preferred, other, image_ids = ordered_pairs(manifest["strict_pairs"])
    missing = [image_id for image_id in image_ids if image_id not in grids]
    if missing:
        raise ValueError("missing cached pre-mask grids for reviewed images: %s"
                         % ", ".join(missing))
    pooled = torch.stack([pool_grid(grids[image_id]) for image_id in image_ids])
    index = {image_id: position for position, image_id in enumerate(image_ids)}
    preferred_idx = torch.tensor([index[i] for i in preferred], dtype=torch.long)
    other_idx = torch.tensor([index[i] for i in other], dtype=torch.long)

    torch.manual_seed(seed)
    channels = pooled.shape[1] // 2
    projection = PooledSceneProjection(channels, values)
    scorer = RankNetScorer(values)
    optimizer = torch.optim.Adam(
        list(projection.parameters()) + list(scorer.parameters()), lr=lr)
    history = []
    interval = max(1, steps // 20)
    for step in range(steps):
        optimizer.zero_grad()
        scene = projection(pooled)              # training: fake uint8
        scores = scorer(fake_uint8(scene))
        loss = pairwise_logistic_loss(scores[preferred_idx], scores[other_idx])
        loss.backward()
        optimizer.step()
        if step % interval == 0 or step + 1 == steps:
            history.append({"step": step, "loss": round(float(loss.detach()), 6)})
    with torch.no_grad():
        eval_scene = (exact_uint8(projection(pooled)).astype(np.float32) / 255.0)
        evaluated = scorer(torch.from_numpy(eval_scene))
        final = float(
            pairwise_logistic_loss(evaluated[preferred_idx], evaluated[other_idx]))
    return {
        "artifact": "trial2_scene_descriptor_ranknet",
        "values": int(values),
        "channels": int(channels),
        "pooled_features": int(2 * channels),
        "training_quantization": "fake_uint8",
        "evaluation_quantization": "exact_uint8",
        "steps": int(steps),
        "lr": float(lr),
        "seed": int(seed),
        "manifest": manifest["path"],
        "review_kind": manifest["review_kind"],
        **{key: manifest[key] for key in ("calibration", "model_revisions", "prompt_version")
           if key in manifest},
        "reviewers": manifest["reviewers"],
        "pair_counts": {
            "strict_ordered": len(preferred),
            "images": len(image_ids),
            "excluded": manifest["excluded"],
        },
        "history": history,
        "final_loss_fake_quant": history[-1]["loss"],
        "final_loss_exact_uint8": round(final, 6),
        "projection_weight": projection.projection.weight.detach().cpu().tolist(),
        "projection_bias": projection.projection.bias.detach().cpu().tolist(),
        "scorer_weight": scorer.scorer.weight.detach().cpu().flatten().tolist(),
        "scorer_bias": float(scorer.scorer.bias.detach()),
        "inputs": "pooled pre-mask grid values only; no wait, service time, "
                  "identity or history",
    }


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--grids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--values", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        artifact = train(load_grids(args.grids), args.manifest,
                         values=args.values, steps=args.steps, lr=args.lr,
                         seed=args.seed)
    except ReviewedPairUnavailable as exc:
        print("BLOCKED: %s" % exc)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("trained %s -> %s" % (artifact["artifact"], args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

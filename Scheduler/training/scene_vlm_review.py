"""Validate and map Trial 2 model judgments to training pairs."""

MODEL_REVISIONS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1": "437f4e28b989cc2d9a16b6767cc930cdf48797ff",
}
REVIEW_KIND = "vlm_scene_pair_review"
SPLITS = {"calibration", "train", "development", "test2", "test3"}
ANSWERS = {"A", "B", "tie", "uncertain"}


def stable_choice(a, b, forward, reverse):
    """Return the mapped stable choice, or None for invalid or unstable answers."""
    first = {"A": a, "B": b, "tie": "tie", "uncertain": "uncertain"}
    second = {"A": b, "B": a, "tie": "tie", "uncertain": "uncertain"}
    if not all(isinstance(answer, str) and answer in ANSWERS
               for answer in (forward, reverse)):
        return None
    x, y = first[forward], second[reverse]
    return x if x == y else None


def training_consensus(manifest):
    """Check provenance and return only same-tier consensus training pairs."""
    if manifest.get("model_revisions") != MODEL_REVISIONS or not manifest.get("prompt_version"):
        raise ValueError("VLM review requires pinned model revisions and prompt version")
    images = manifest["images"]
    group_splits = {}
    for image_id, image in images.items():
        split = image["split"]
        if split not in SPLITS or not image_id or image_id in {"tie", "uncertain"}:
            raise ValueError("invalid image ID or review split")
        for field in ("event", "flight", "near_duplicate"):
            group = image[field]
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f"missing {field} group for {image_id}")
            if group_splits.setdefault((field, group), split) != split:
                raise ValueError(f"{field} group overlaps review splits: {group}")
    accepted = []
    seen = set()
    calibration = []
    for pair in manifest["pairs"]:
        a, b = pair["a"], pair["b"]
        if a == b:
            raise ValueError("self-pairs are reserved for separate reviewer controls")
        key = frozenset((a, b))
        if key in seen:
            raise ValueError("duplicate unordered review pair")
        seen.add(key)
        left, right = images[a], images[b]
        if left["split"] != right["split"]:
            raise ValueError("review pair crosses splits")
        tier = left["spi_tier"]
        if (type(tier) is not int or tier not in range(4)
                or type(right["spi_tier"]) is not int or tier != right["spi_tier"]):
            raise ValueError("review pair must have the same predicted SPI tier")
        if left["split"] not in {"train", "calibration"}:
            continue
        votes = pair["votes"]
        if set(votes) != set(MODEL_REVISIONS):
            raise ValueError("review pair requires both pinned reviewers")
        responses = [(votes[model]["forward"], votes[model]["reverse"]) for model in MODEL_REVISIONS]
        choices = [stable_choice(a, b, *response) for response in responses]
        if left["split"] == "calibration":
            calibration.append([(all(isinstance(answer, str) and answer in ANSWERS
                                     for answer in response),
                                 choice, choice in (a, b))
                                for response, choice in zip(responses, choices)])
            continue
        if choices[0] == choices[1] and choices[0] in (a, b):
            accepted.append({"a": a, "b": b, "grade_a": int(choices[0] == a),
                             "grade_b": int(choices[0] == b), "status": "strict",
                             "same_tier": True, "reviewer": "Qwen+Nemotron consensus"})
    if not calibration:
        raise ValueError("VLM manifest requires a separate calibration cohort")
    metrics = {}
    for model, decisions in zip(MODEL_REVISIONS, zip(*calibration)):
        valid = sum(valid for valid, _, _ in decisions)
        stable = sum(choice is not None for _, choice, _ in decisions)
        strict = sum(strict for _, _, strict in decisions)
        metrics[model] = {"valid": valid, "stable": stable, "strict": strict}
        if not valid or stable / valid < 0.90 or strict / len(calibration) < 0.50:
            raise ValueError(f"VLM calibration thresholds not met for {model}")
    overlap = [(left[1], right[1]) for left, right in calibration if left[2] and right[2]]
    agreement = sum(left == right for left, right in overlap)
    if not overlap or agreement / len(overlap) < 0.80:
        raise ValueError("VLM calibration agreement threshold not met")
    if not accepted:
        raise ValueError("VLM manifest contains no stable same-tier training consensus")
    return accepted, {"pairs": len(calibration), "models": metrics,
                      "joint_strict": len(overlap), "joint_agreement": agreement}

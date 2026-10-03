"""Replay saved whole-latent order through the existing checkpoint codec."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace", "images", "payload-costs", "model-dir", "upstream-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--snr-db", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    if trace.get("schema") != "latent-schedule/v1":
        raise ValueError("Expected a canonical latent schedule")
    with args.images.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    images = {}
    for row in rows:
        if row["image_id"] in images:
            raise ValueError("Duplicate image identity")
        path = Path(row["image_path"])
        images[row["image_id"]] = path if path.is_absolute() else args.images.parent / path
    jobs = {job["job_id"]: job for job in trace["jobs"]}
    selected = [jobs[row["job_id"]] for row in trace["decision_trace"]]
    if not selected:
        raise ValueError("Nothing was selected for replay")
    with args.payload_costs.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    costs = {row["image_id"]: row for row in rows}
    if len(costs) != len(rows):
        raise ValueError("Duplicate payload identity")
    for job in selected:
        cost = costs[job["image_id"]]
        if (job["resource_unit"] != "complex_channel_uses" or
                job["resource_units"] != int(cost["channel_uses"]) or
                job["payload_provenance"] != "swinjscc:" + cost["checkpoint_sha256"]):
            raise ValueError("Scheduled payload does not match codec profile")
    from EncDec.integration.scheduler_replay import replay_selected_jobs
    print(json.dumps({"stage": "replay_start", "latents": len(selected)}), flush=True)
    output = replay_selected_jobs([job["image_id"] for job in selected], images,
        args.payload_costs, args.output, model_dir=args.model_dir,
        upstream_root=args.upstream_root, snr_db=args.snr_db, seed=args.seed, device=args.device)
    print(json.dumps({"stage": "replay_complete", "metrics": str(output)}), flush=True)


if __name__ == "__main__":
    main()

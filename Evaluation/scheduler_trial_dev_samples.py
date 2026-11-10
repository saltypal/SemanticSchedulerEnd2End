"""Fetch a bounded, label-balanced LADI validation cohort for development only."""

import argparse
import json
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DATASET = "MITLL/LADI-v2-dataset"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    counts = {False: 0, True: 0}
    manifest = []
    with requests.Session() as session:
        retries = Retry(total=3, backoff_factor=0.7, status_forcelist=(429, 500, 502, 503, 504))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        info = session.get(f"https://huggingface.co/api/datasets/{DATASET}", timeout=30)
        info.raise_for_status()
        revision = info.json()["sha"]
        for offset in range(0, 600, 25):
            response = session.get("https://datasets-server.huggingface.co/rows", params={
                "dataset": DATASET, "config": "default", "split": "validation",
                "offset": offset, "length": 25,
            }, timeout=45)
            response.raise_for_status()
            for item in response.json()["rows"]:
                row = item["row"]
                positive = row["roads_damage"]
                if row["roads_any"] is not True or type(positive) is not bool or counts[positive] >= 12:
                    continue
                key = f"ladi-val-{item['row_idx']:04d}"
                response = session.get(row["image"]["src"], timeout=45)
                response.raise_for_status()
                path = args.output / f"{key}.jpg"
                path.write_bytes(response.content)
                manifest.append({
                    "item_id": key, "row_idx": item["row_idx"], "image_path": path.name,
                    "positive_labels": [k for k, v in row.items() if v is True],
                    "dataset": DATASET, "source_split": "validation",
                    "dataset_head_observed": revision, "image_delivery": "datasets-server image asset",
                })
                counts[positive] += 1
            if min(counts.values()) == 12:
                break
    (args.output / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(counts)
    if min(counts.values()) < 12:
        raise SystemExit("Development cohort did not reach the planned class quota")


if __name__ == "__main__":
    main()

"""Build a deterministic candidate list from the public LADI v2a label CSVs."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit


EVENT_SPLITS = {
    "CAP_-_Hur_Ian_Sept_2022": "train",
    "CAP_-_TS_Fiona_Sept_2022": "train",
    "CAP_-_KY_Flooding_July_2022": "calibration",
    "CAP_-_MT_Floods_2022": "calibration",
    "CAP_-_TS_Nicole_Nov_2022": "development",
    "CAP_-_AR_Storms_APR_2023": "test2",
    "CAP_-_CA_Storms_Jan_2023": "test2",
    "CAP_-_HI_Wildfires_Aug_2023": "test2",
    "CAP_-_Hur_Idalia_Aug_2023": "test3",
    "CAP_-_MS_Tornadoes_MAR_2023": "test3",
    "CAP_-_PA_Storms_Aug_2023": "test3",
    "CAP_-_VT_Flooding_Jul_2023": "test3",
}


def build_candidates(csv_paths, per_stratum=5):
    strata = defaultdict(dict)
    mission_splits = {}
    for path in csv_paths:
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                url = row["url"]
                address = urlsplit(url)
                parts = unquote(address.path).strip("/").split("/")
                if address.scheme != "https" or address.netloc != "fema-cap-imagery.s3.amazonaws.com":
                    continue
                if len(parts) != 6 or parts[0] != "Images" or parts[2] != "Source":
                    continue
                event, mission, sortie = parts[1], parts[3], parts[4]
                if event not in EVENT_SPLITS or not mission or not sortie:
                    continue
                split = EVENT_SPLITS[event]
                if mission_splits.setdefault(mission, split) != split:
                    raise ValueError(f"publisher mission crosses Trial 2 splits: {mission}")
                tagged = any(row[key] == "1" for key in
                             ("flooding_structures", "buildings_affected_or_greater", "debris_any"))
                strata[event, tagged][url] = {
                    "publisher_url": url, "event": event, "mission": mission,
                    "flight": f"{mission}/{sortie}", "split": split,
                    "sampling_impact_tag": tagged,
                }
    rows = []
    for key in sorted(strata):
        candidates = strata[key]
        urls = sorted(candidates, key=lambda url: hashlib.sha256(url.encode()).digest())
        rows.extend(candidates[url] for url in urls[:per_stratum])
    if {row["split"] for row in rows} != set(EVENT_SPLITS.values()):
        raise ValueError("candidate list is missing a Trial 2 split")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = build_candidates(args.csv)
    receipt = {
        "scope": "Candidates only. Mission/sortie inferred from publisher URL. "
                 "Image verification, duplicate screening and SPI are still required.",
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in args.csv},
        "images": rows,
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2)


if __name__ == "__main__":
    main()

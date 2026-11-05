import concurrent.futures
import json
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest


OWNER = "satyapaladugu"
SLUG = "crisismmd1000-spi-labeling"
VERSION = 6
KERNEL_ID = 141416127
SOURCE_SHA256 = "8AD229ADB25034DDB0673EC5A26D88C8AD8E33CF5A3D714BB106E3FF3DF97C85"
PROJECT = Path(r"D:\Bunker\BaseCamp\WirelessCommunication\SemanticCommunication\SemanticSchedulerEnd2End")
RESULTS = PROJECT / "experiments/full_medic_labeling/runs/v6_results"
RUN_DIR = Path(__file__).resolve().parent
PATHS_FILE = RUN_DIR / "v6_candidate_artifact_paths.json"
OUTPUT = RUN_DIR / "v6_model_size_plan.json"
HARD_CAP = 5 * 1024**3


def probe(path):
    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug = OWNER
    request.kernel_slug = SLUG
    request.file_path = path
    request.version_number = VERSION
    try:
        response = client.kernels.kernels_api_client.download_kernel_output(request)
        result = {
            "path": path,
            "status_code": response.status_code,
            "content_length": int(response.headers["Content-Length"])
            if response.headers.get("Content-Length", "").isdigit()
            else None,
        }
        response.close()
        return result
    except Exception as error:
        response = getattr(error, "response", None)
        return {
            "path": path,
            "status_code": getattr(response, "status_code", None),
            "error_type": type(error).__name__,
        }


def main():
    paths = json.loads(PATHS_FILE.read_text(encoding="utf-8"))
    if any("/checkpoints/" in path for path in paths):
        raise SystemExit("Refusing to retrieve checkpoint duplicates")
    existing = sum(path.stat().st_size for path in RESULTS.rglob("*") if path.is_file())
    print(f"MODEL SIZE PROBE START count={len(paths)} existing_bytes={existing}", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index, result in enumerate(pool.map(probe, paths), 1):
            results.append(result)
            if index % 40 == 0:
                print(f"MODEL SIZE PROBE progress={index}/{len(paths)}", flush=True)
    planned = sum(item.get("content_length") or 0 for item in results)
    missing = [item for item in results if item.get("status_code") != 200 or item.get("content_length") is None]
    plan = {
        "kernel_id": KERNEL_ID,
        "kernel": f"{OWNER}/{SLUG}/{VERSION}",
        "source_sha256": SOURCE_SHA256,
        "probed_utc": datetime.now(timezone.utc).isoformat(),
        "hard_cap_bytes": HARD_CAP,
        "existing_results_bytes": existing,
        "planned_model_transfer_bytes": planned,
        "remaining_bytes_before_model_transfer": HARD_CAP - existing,
        "within_hard_cap": planned <= HARD_CAP - existing,
        "missing_or_unprobed_count": len(missing),
        "missing_or_unprobed": missing,
        "files": results,
    }
    OUTPUT.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        f"MODEL SIZE PROBE END planned_bytes={planned} remaining={HARD_CAP-existing} "
        f"within_cap={plan['within_hard_cap']} missing={len(missing)} receipt={OUTPUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()

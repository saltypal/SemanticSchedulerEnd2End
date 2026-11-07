from pathlib import Path
import json
from datetime import datetime, timezone

repo = Path.cwd()
ops = repo / "experiments/spi_four_class/runs/ops_preflight_20261027"
stage = repo / "experiments/spi_four_class/runs/20261027_spi_evidence_v3b/submission"
package = json.loads((stage / "PACKAGE.json").read_text())
validation = json.loads((repo / "experiments/spi_four_class/runs/affected_parser_checks_20261027/v3b_package_validation.json").read_text())
record = {
    "recorded_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": package["run_id"],
    "current_ref": package["submission_current_ref"],
    "kernel_numeric_id": package["kernel_numeric_id"],
    "prior_version": 2,
    "prior_status": "ERROR",
    "expected_new_version": 3,
    "title_restore_expected_ref": package["title_restore_expected_ref"],
    "input": package["input"],
    "input_version": 1,
    "input_status": "ready",
    "source_zip": str((stage / "source.zip").relative_to(repo)),
    "source_zip_sha256": package["source_sha256"],
    "timeout_seconds": package["server_seconds"],
    "aggregate_gpu_hour_cap": package["aggregate_gpu_hour_cap"],
    "accelerator": "NvidiaTeslaT4",
    "affected_tests": "26 passed in 14.35s; no source drift",
    "offline_replay": "10 parsed, exact wrapper normalization only, no contradictions; Low8 Medium1 High0 Critical1",
    "prelaunch_kernel_status": (ops / "pre_v3b_kernel_status.txt").read_text().strip(),
    "quota": json.loads((ops / "pre_v3b_quota.json").read_text(encoding="utf-8-sig").split("\n", 1)[-1]),
    "package_validation": validation,
    "expected_output_root": "/kaggle/working/experiments/spi_four_class/runs/20261027_spi_evidence_v3b/",
    "submit_command": "kaggle kernels push -p experiments/spi_four_class/runs/20261027_spi_evidence_v3b/submission -t 7200 --accelerator NvidiaTeslaT4",
}
(ops / "v3b_prelaunch.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

section = """

### v2 terminal evidence and v3b retry prelaunch

- Version 2 (`20261027_spi_evidence_v3a`) ended ERROR in cell 2 after the
  10-image visual-transport preflight; cell 3 did not run. `campaign/status.json`
  is incomplete, zero labels and no four-class model. The root cause was strict
  parsing of complete JSON replies inside Markdown fences: all 10 were initially
  marked `parse_failed`. The current parser accepts only the exact complete
  fence wrapper; offline replay parsed all 10 with claims/evidence unchanged
  (8 Low, 1 Medium, 1 Critical, 0 High), no invalid response or nested-claim
  contradiction, and no model/prompt/class change. Request receipts confirm one
  image per request, image-bearing rendered text, only the instruction payload,
  evidence protocol, and thinking OFF. Generic legacy provenance
  `mode=annotation_assisted_visible_json` remains ambiguous metadata; actual
  request payloads show no annotations, and raw provenance was preserved.
- Retrieval was pinned to current-ref `/2`. 186 files retrieved; campaign direct
  files and all 45 checksum entries match; recovery ZIP SHA-256
  `386323ba5ca009377bbbcb33d7da026a113185d912c75edc03d8c79bef36f90d`, CRC
  valid, all 45 members match, and 108 source files match the runtime manifest.
  Kaggle CLI returned 1 after files arrived while printing a Windows charmap
  error; local integrity passed. Runtime confirms Python 3.13.15, two Tesla T4,
  CUDA 12.8. Cell logs and executed notebook were retrieved. Quota after v2:
  3.35 h used / 26.65 h remaining. Receipts and extracted recovery are under
  `experiments/spi_four_class/runs/20261027_spi_evidence_v3a/retrieved_version2/`.
- Affected parser tests passed 26/26 (`test_pipeline.py` and `test_spi.py`),
  with no source drift. Frozen v3b package SHA-256
  `bf06d6779e0bfcdf9f89c78218b60cebcfe22d996ae4fda999bbd77aaec3c85e`; embedded
  archive matches, 105 canonical source hashes match and three documented
  namespace stubs match their exact builder literal. Notebook schema and cells
  pass. Metadata targets numeric kernel ID `141089219`, uses the current alias
  for update, and restores title `BetterScheduler SPI 20261027finalc` so the
  original ref should return. Input remains private v1 READY. Prior v2 is
  terminal ERROR; v3b is authorized as version 3 on the same object, with
  7,200 seconds and a 4 aggregate GPU-hour cap. Prelaunch receipt:
  `experiments/spi_four_class/runs/ops_preflight_20261027/v3b_prelaunch.json`.
"""
with (repo / "parameters/KAGGLE_RUNS.md").open("a", encoding="utf-8") as handle:
    handle.write(section)
print(record["run_id"], record["source_zip_sha256"], "prelaunch-recorded")

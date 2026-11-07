import hashlib
import json
import pathlib

run = pathlib.Path(__file__).resolve().parent
remote = run / "retrieved/experiments/stream_scheduler/runs/repair-followup-20261101T033409Z"
manifest = json.loads((remote / "artifact_manifest.json").read_text())
checked, mismatches, missing = [], [], []
for relative, expected in manifest.items():
    path = remote / relative
    if not path.is_file():
        missing.append(relative)
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    checked.append(relative)
    if actual != expected:
        mismatches.append({"path": relative, "expected": expected, "actual": actual})

validation = json.loads((remote / "results/validation.json").read_text())
responses_path = remote / "results/teacher/responses.jsonl"
responses = [json.loads(line) for line in responses_path.read_text().splitlines() if line.strip()]
result = {
    "kernel": "satyapaladuguplus/semantic-decoder-scheduler-20261030/6",
    "remote_run": "repair-followup-20261101T033409Z",
    "platform_status": "COMPLETE",
    "manifest_entries": len(manifest),
    "retrieved_and_hash_verified": len(checked),
    "verified_paths": checked,
    "selectively_not_retrieved": len(missing),
    "hash_mismatch_count": len(mismatches),
    "hash_mismatches": mismatches,
    "validation": validation,
    "teacher_responses": len(responses),
    "teacher_request_ids": [row.get("id") for row in responses],
    "teacher_statuses": [row.get("status") for row in responses],
    "teacher_provenance_sha256": hashlib.sha256(
        (remote / "results/teacher/provenance.json").read_bytes()
    ).hexdigest(),
    "status": "verified" if checked and not mismatches else "failed",
}
(run / "retrieval_verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({
    "status": result["status"],
    "manifest_entries": result["manifest_entries"],
    "verified_retrieved": result["retrieved_and_hash_verified"],
    "hash_mismatch_count": result["hash_mismatch_count"],
    "teacher_responses": result["teacher_responses"],
    "teacher_request_ids": result["teacher_request_ids"],
    "validation": result["validation"],
}, indent=2))
raise SystemExit(bool(mismatches or not checked))

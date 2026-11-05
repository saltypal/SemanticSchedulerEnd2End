import hashlib
import json
import pathlib

run = pathlib.Path(__file__).resolve().parent
remote = run / "retrieved/experiments/stream_scheduler/runs/repair-validation-20261101T030831Z"
manifest = json.loads((remote / "artifact_manifest.json").read_text())
checked = []
missing = []
mismatches = []
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
    "kernel": "satyapaladuguplus/semantic-decoder-scheduler-20261030/5",
    "remote_run": "repair-validation-20261101T030831Z",
    "terminal_platform_status": "COMPLETE",
    "remote_manifest_entries": len(manifest),
    "retrieved_verified_count": len(checked),
    "verified_paths": checked,
    "missing_by_selective_retrieval": len(missing),
    "hash_mismatch_count": len(mismatches),
    "hash_mismatches": mismatches,
    "learner_and_channel_status": validation,
    "teacher_response_count": len(responses),
    "teacher_task_counts": {
        task: sum(row.get("task") == task for row in responses)
        for task in sorted({row.get("task") for row in responses})
    },
    "teacher_model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "teacher_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "teacher_claim_scope": "format/control gate only; not emergency-priority accuracy",
    "status": "verified" if checked and not mismatches else "failed",
}
(run / "retrieval_verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({
    "status": result["status"],
    "remote_manifest_entries": result["remote_manifest_entries"],
    "retrieved_verified_count": result["retrieved_verified_count"],
    "hash_mismatch_count": result["hash_mismatch_count"],
    "teacher_response_count": result["teacher_response_count"],
    "teacher_task_counts": result["teacher_task_counts"],
}, indent=2))
raise SystemExit(bool(mismatches or not checked))

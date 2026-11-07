from pathlib import Path
import hashlib, json, py_compile, sys

root = Path("/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End")
run = root / "experiments/stream_scheduler/runs/full-v22-launch-20261031T003706Z"
manifest = json.loads((run / "source_manifest.json").read_text())
canonical = root / "experiments/stream_scheduler/kaggle/Semantic_Decoder_Scheduler.ipynb"
prepared = run / "prepared_notebook.ipynb"
cfg = json.loads((run / "launch_config.json").read_text())
meta = json.loads((run / "kernel-metadata.json").read_text())

mismatches = []
compile_errors = []
for rel, expected in manifest.items():
    path = root / rel
    if not path.is_file():
        mismatches.append({"path": rel, "error": "missing"})
        continue
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != expected:
        mismatches.append({"path": rel, "expected": expected, "actual": got})
    if path.suffix == ".py":
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except Exception as exc:
            compile_errors.append({"path": rel, "error": repr(exc)})

nb = json.loads(prepared.read_text(encoding="utf-8"))
nb_cells = []
for i, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        try:
            compile(src, f"notebook-cell-{i}", "exec")
        except Exception as exc:
            compile_errors.append({"path": f"prepared_notebook.ipynb:cell-{i}", "error": repr(exc)})
        nb_cells.append({"index": i, "length": len(src), "sha256": hashlib.sha256(src.encode()).hexdigest()})

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

out = {
    "run_id": run.name,
    "manifest_source_count": len(manifest),
    "manifest_missing_or_hash_mismatch": mismatches,
    "python_source_compile_errors": compile_errors,
    "compiled_manifest_python_files": sum(1 for p in manifest if p.endswith(".py")),
    "notebook_code_cells": len(nb_cells),
    "notebook_code_cell_details": nb_cells,
    "prepared_notebook_sha256": sha(prepared),
    "canonical_notebook_sha256": sha(canonical),
    "launch_config_expected_notebook_sha256": cfg.get("notebook_sha256"),
    "canonical_equals_prepared": canonical.read_bytes() == prepared.read_bytes(),
    "metadata": {k: meta.get(k) for k in ("id", "is_private", "enable_gpu", "enable_internet", "kernel_sources")},
    "launch_config": cfg,
    "status": "PASS" if not mismatches and not compile_errors and len(manifest) == 114 and canonical.read_bytes() == prepared.read_bytes() and sha(canonical) == cfg.get("notebook_sha256") else "FAIL",
}
(run / "v4_static_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))

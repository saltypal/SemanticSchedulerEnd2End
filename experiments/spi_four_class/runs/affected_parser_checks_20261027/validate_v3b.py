from pathlib import Path
import ast
import base64
import hashlib
import json
import re
import zipfile
import nbformat

repo = Path.cwd()
stage = repo / "experiments/spi_four_class/runs/20261027_spi_evidence_v3b/submission"
package = json.loads((stage / "PACKAGE.json").read_text())
metadata = json.loads((stage / "kernel-metadata.json").read_text())
notebook = nbformat.read(stage / "notebook.ipynb", as_version=4)
nbformat.validate(notebook)
compiled_cells = []
code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"<cell-{index}>", "exec")
        compiled_cells.append(index)

archive_bytes = (stage / "source.zip").read_bytes()
archive_hash = hashlib.sha256(archive_bytes).hexdigest()
embedded_match = re.search(r"base64\.b64decode\(['\"]([^'\"]+)['\"]\)", code)
embedded_bytes = base64.b64decode(embedded_match.group(1)) if embedded_match else b""

with zipfile.ZipFile(stage / "source.zip") as archive:
    crc_error = archive.testzip()
    manifest = json.loads(archive.read("SOURCE_MANIFEST.json"))
    member_mismatches = []
    canonical_mismatches = []
    missing_canonical = []
    for name, expected in manifest["files"].items():
        content = archive.read(name)
        if hashlib.sha256(content).hexdigest() != expected:
            member_mismatches.append(name)
        local = repo / name
        if not local.is_file():
            missing_canonical.append(name)
        elif hashlib.sha256(local.read_bytes()).digest() != hashlib.sha256(content).digest():
            canonical_mismatches.append(name)

    builder = ast.parse((repo / "parameters/build_annotation_notebook.py").read_text())
    stub_paths = []
    stub_content = None
    for node in ast.walk(builder):
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == "name" and isinstance(node.iter, (ast.Tuple, ast.List))
                and all(isinstance(item, ast.Constant) and isinstance(item.value, str)
                        for item in node.iter.elts)):
            candidates = [item.value for item in node.iter.elts]
            if candidates and all(item.endswith("__init__.py") for item in candidates):
                stub_paths = candidates
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, bytes)):
            stub_content = node.value.value
    stub_mismatches = [name for name in stub_paths if archive.read(name) != stub_content]
    v3b_files = dict(manifest["files"])
    v3b_source = {name: archive.read(name) for name in v3b_files}

v3a_stage = repo / "experiments/spi_four_class/runs/20261027_spi_evidence_v3a/submission_final/source.zip"
with zipfile.ZipFile(v3a_stage) as previous_archive:
    previous_manifest = json.loads(previous_archive.read("SOURCE_MANIFEST.json"))
previous_files = previous_manifest["files"]
changed_source_paths = sorted(
    name for name, digest in v3b_files.items()
    if previous_files.get(name) != digest
)
hash_dir = repo / "experiments/spi_four_class/runs/affected_parser_checks_20261027"
hash_before = json.loads((hash_dir / "source_hashes_before.json").read_text(encoding="utf-8-sig"))
hash_after = json.loads((hash_dir / "source_hashes_after.json").read_text(encoding="utf-8-sig"))
before_by_path = {row["path"]: row["sha256"] for row in hash_before}
source_drift_during_tests = sorted(
    row["path"] for row in hash_after
    if before_by_path.get(row["path"]) != row["sha256"]
)

required = [
    "project_compat.py",
    "parameters/spi_protocol.py",
    "parameters/spi_pipeline.py",
    "parameters/spi_teachers.py",
    "parameters/spi_model_selection.py",
    "parameters/spi_final_campaign.py",
]
member_names = set(archive.namelist())

summary = {
    "run_id": package.get("run_id"),
    "source_zip_sha256": archive_hash,
    "package_hash_matches": archive_hash == package.get("source_sha256"),
    "expected_hash_matches": archive_hash == "bf06d6779e0bfcdf9f89c78218b60cebcfe22d996ae4fda999bbd77aaec3c85e",
    "embedded_archive_matches": embedded_bytes == archive_bytes,
    "zip_crc": "ok" if crc_error is None else crc_error,
    "manifest_file_count": len(manifest["files"]),
    "changed_source_paths_vs_v3a": changed_source_paths,
    "source_drift_during_affected_tests": source_drift_during_tests,
    "manifest_hash_mismatches": member_mismatches,
    "canonical_non_stub_mismatches": [name for name in canonical_mismatches if name not in stub_paths],
    "canonical_missing": missing_canonical,
    "namespace_stubs": stub_paths,
    "namespace_stubs_match_builder": not stub_mismatches,
    "namespace_stub_mismatches": stub_mismatches,
    "required_source_presence": {name: name in member_names for name in required},
    "nbformat": "valid",
    "compiled_cells": compiled_cells,
    "kernel_metadata_id": metadata.get("id"),
    "kernel_metadata_id_no": metadata.get("id_no"),
    "package_numeric_id": package.get("kernel_numeric_id"),
    "title": metadata.get("title"),
    "dataset_sources": metadata.get("dataset_sources"),
    "package_input": package.get("input"),
    "server_seconds": package.get("server_seconds"),
    "aggregate_gpu_hour_cap": package.get("aggregate_gpu_hour_cap"),
    "runtime_receipt_written_and_copied": all(value in code for value in (
        "runtime.json", "shutil.copy2('/kaggle/working/runtime.json'",
    )),
}
out = Path("experiments/spi_four_class/runs/affected_parser_checks_20261027/v3b_package_validation.json")
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))

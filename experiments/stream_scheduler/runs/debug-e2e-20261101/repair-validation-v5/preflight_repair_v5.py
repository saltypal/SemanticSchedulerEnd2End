import ast
import base64
import hashlib
import json
import pathlib
import zipfile
from io import BytesIO

ROOT = pathlib.Path(__file__).resolve().parents[5]
RUN = pathlib.Path(__file__).resolve().parent
NOTEBOOK = ROOT / "experiments/stream_scheduler/kaggle/Semantic_Decoder_Scheduler.ipynb"
METADATA = ROOT / "experiments/stream_scheduler/kaggle/kernel-metadata.json"
MANIFEST = ROOT / "experiments/stream_scheduler/runs/debug-e2e-20261101/repair_notebook_manifest.json"
errors = []
notebook_bytes = NOTEBOOK.read_bytes()
manifest = json.loads(MANIFEST.read_text())
nb = json.loads(notebook_bytes)
if hashlib.sha256(notebook_bytes).hexdigest() != manifest["notebook_sha256"]:
    errors.append("notebook SHA does not match repair manifest")
if len(notebook_bytes) >= 1_000_000:
    errors.append(f"notebook too large for Kaggle source limit: {len(notebook_bytes)}")
if hashlib.sha256(METADATA.read_bytes()).hexdigest() != "930b9fb6d442ccc9bf81659ed91474c60049b54bd41f8222e74dc91a60bfce51":
    errors.append("kernel metadata SHA changed")
code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
if len(code_cells) != 5:
    errors.append(f"expected 5 code cells, got {len(code_cells)}")
script = "".join(code_cells[1]["source"])
tree = ast.parse(script)
blob = None
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "b64decode" and node.args and isinstance(node.args[0], ast.Constant):
        blob = base64.b64decode(node.args[0].value)
        break
if blob is None:
    errors.append("could not locate embedded ZIP bundle")
else:
    with zipfile.ZipFile(BytesIO(blob)) as archive:
        embedded = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if not name.endswith("/")}
        if len(embedded) != manifest["bundle_files"]:
            errors.append(f"bundle member count mismatch: {len(embedded)} != {manifest['bundle_files']}")
        for rel, expected in manifest["source_hashes"].items():
            if embedded.get(rel) != expected:
                errors.append(f"embedded source mismatch: {rel}")
        for rel, expected in manifest["data_hashes"].items():
            if embedded.get(rel) != expected:
                errors.append(f"embedded data mismatch: {rel}")
        python_files = [name for name in embedded if name.endswith(".py")]
        for name in python_files:
            try:
                ast.parse(archive.read(name), filename=name)
            except SyntaxError as exc:
                errors.append(f"Python syntax error {name}: {exc}")
result = {
    "notebook_bytes": len(notebook_bytes),
    "notebook_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
    "notebook_manifest_sha256": manifest["notebook_sha256"],
    "kernel_metadata_sha256": hashlib.sha256(METADATA.read_bytes()).hexdigest(),
    "source_count": len(manifest["source_hashes"]),
    "data_count": len(manifest["data_hashes"]),
    "bundle_members": len(embedded) if blob else 0,
    "python_sources_parsed": len(python_files) if blob else 0,
    "errors": errors,
    "status": "passed" if not errors else "failed",
}
(RUN / "preflight_repair_v5.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(bool(errors))

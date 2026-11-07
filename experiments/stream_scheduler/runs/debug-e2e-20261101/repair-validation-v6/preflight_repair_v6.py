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
notebook_bytes = NOTEBOOK.read_bytes()
manifest = json.loads(MANIFEST.read_text())
errors = []
if hashlib.sha256(notebook_bytes).hexdigest() != manifest["notebook_sha256"]:
    errors.append("canonical notebook SHA does not match manifest")
if len(notebook_bytes) >= 1_000_000:
    errors.append(f"notebook exceeds Kaggle source limit: {len(notebook_bytes)} bytes")
expected_metadata = "930b9fb6d442ccc9bf81659ed91474c60049b54bd41f8222e74dc91a60bfce51"
if hashlib.sha256(METADATA.read_bytes()).hexdigest() != expected_metadata:
    errors.append("metadata SHA changed")
metadata = json.loads(METADATA.read_text())
if metadata.get("id") != "satyapaladuguplus/semantic-decoder-scheduler-20261030" or not metadata.get("is_private") or not metadata.get("enable_gpu") or not metadata.get("enable_internet"):
    errors.append("Kaggle identity or launch settings differ from approved configuration")
notebook = json.loads(notebook_bytes)
cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
if len(cells) != 5:
    errors.append(f"expected 5 code cells; found {len(cells)}")
script = "".join(cells[1]["source"])
blob = None
for node in ast.walk(ast.parse(script)):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "b64decode" and node.args and isinstance(node.args[0], ast.Constant):
        blob = base64.b64decode(node.args[0].value)
        break
embedded = {}
python_count = 0
actual_ids = []
prior_provenance = {}
if blob is None:
    errors.append("embedded source ZIP not found")
else:
    with zipfile.ZipFile(BytesIO(blob)) as archive:
        embedded = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if not name.endswith("/")
        }
        if len(embedded) != manifest["bundle_files"]:
            errors.append(f"member count differs: {len(embedded)} != {manifest['bundle_files']}")
        expected = {**manifest["source_hashes"], **manifest["data_hashes"]}
        for relative, sha in expected.items():
            if embedded.get(relative) != sha:
                errors.append(f"embedded hash mismatch: {relative}")
        python_files = [name for name in embedded if name.endswith(".py")]
        python_count = len(python_files)
        for name in python_files:
            try:
                ast.parse(archive.read(name), filename=name)
            except SyntaxError as exc:
                errors.append(f"syntax error in {name}: {exc}")
        request_name = "repair_inputs/technical_requests.jsonl"
        if request_name in embedded:
            requests = [
                json.loads(line)
                for line in archive.read(request_name).decode("utf-8").splitlines()
                if line.strip()
            ]
            expected_ids = [
                "train-1-0",
                "train-1-1",
                "prefix-train-0-0",
                "prefix-train-0-1",
                "replay-0",
            ]
            actual_ids = [item.get("id") for item in requests]
            if actual_ids != expected_ids:
                errors.append(f"technical request IDs/order mismatch: {actual_ids}")
        prior_provenance = json.loads(archive.read("repair_inputs/prior_provenance.json"))
        if prior_provenance.get("model_id") != "Qwen/Qwen2.5-VL-7B-Instruct":
            errors.append("prior teacher model ID mismatch")
        if prior_provenance.get("revision") != "cc594898137f460bfe9f0759e9844b3ce807cfb5":
            errors.append("prior teacher revision mismatch")
        if prior_provenance.get("quantization") != "NF4_double_FP16":
            errors.append("prior teacher quantization mismatch")
        for relative, sha in manifest["source_hashes"].items():
            path = ROOT / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                errors.append(f"local source no longer matches frozen manifest: {relative}")
result = {
    "notebook_bytes": len(notebook_bytes),
    "notebook_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
    "manifest_notebook_sha256": manifest["notebook_sha256"],
    "metadata_sha256": hashlib.sha256(METADATA.read_bytes()).hexdigest(),
    "bundle_files_expected": manifest["bundle_files"],
    "bundle_files_embedded": len(embedded),
    "source_files": len(manifest["source_hashes"]),
    "data_files": len(manifest["data_hashes"]),
    "python_files_ast_parsed": python_count,
    "technical_request_ids": actual_ids if blob else [],
    "prior_teacher_model": prior_provenance.get("model_id") if blob else None,
    "prior_teacher_revision": prior_provenance.get("revision") if blob else None,
    "prior_teacher_quantization": prior_provenance.get("quantization") if blob else None,
    "errors": errors,
    "status": "passed" if not errors else "failed",
}
(RUN / "preflight_repair_v6.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(bool(errors))

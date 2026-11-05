import ast, base64, hashlib, io, json, zipfile
from pathlib import Path
root=Path("/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End")
run=root/"experiments/stream_scheduler/runs/full-v22-launch-20261031T003706Z"
nb=json.loads((root/"experiments/stream_scheduler/kaggle/Semantic_Decoder_Scheduler.ipynb").read_text())
manifest=json.loads((run/"source_manifest.json").read_text())
cell="".join(nb["cells"][2]["source"])
tree=ast.parse(cell)
blob_literal=None
for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="b64decode" and node.args:
        try:
            value=ast.literal_eval(node.args[0])
        except Exception:
            continue
        if isinstance(value,str) and value.startswith("UEsDB"):
            blob_literal=value
            break
if blob_literal is None:
    raise SystemExit("No embedded source ZIP literal found")
blob=base64.b64decode(blob_literal,validate=True)
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    names=z.namelist()
    hashes={name:hashlib.sha256(z.read(name)).hexdigest() for name in names}
missing=sorted(set(manifest)-set(hashes))
extra=sorted(set(hashes)-set(manifest))
mismatch=sorted(k for k in set(manifest)&set(hashes) if manifest[k]!=hashes[k])
out={"embedded_zip_bytes":len(blob),"embedded_file_count":len(names),"manifest_count":len(manifest),"missing":missing,"extra":extra,"hash_mismatches":mismatch,"status":"PASS" if not missing and not extra and not mismatch else "FAIL"}
(run/"v4_embedded_bundle_audit.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

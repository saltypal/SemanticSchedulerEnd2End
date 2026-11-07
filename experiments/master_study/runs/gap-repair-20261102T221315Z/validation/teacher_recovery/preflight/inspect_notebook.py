import ast,base64,hashlib,json,pathlib,zipfile,io,sys,collections
ROOT=pathlib.Path("/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End")
book=ROOT/"experiments/master_study/gap_recovery/WSL_Teacher_Recovery.ipynb"
manifest_path=ROOT/"experiments/master_study/gap_recovery/manifest.json"
meta_path=ROOT/"experiments/master_study/gap_recovery/kernel-metadata.json"
expected_notebook="2fdea70f7df416957dadd91cd77ce5ac0bc0ca91f80a4726388e99edf830d30c"
expected_capsule="a824d45cb146d850e709a67d29ded1aa2fa8dd3ca1db8248a82a4548fa3291e8"
b=json.loads(book.read_text());m=json.loads(manifest_path.read_text());meta=json.loads(meta_path.read_text())
actual_nb=hashlib.sha256(book.read_bytes()).hexdigest()
print("notebook_sha256",actual_nb,"expected",expected_notebook,"match",actual_nb==expected_notebook)
if actual_nb!=expected_notebook: sys.exit("notebook hash mismatch")
if m.get("notebook_sha256")!=expected_notebook: sys.exit("manifest notebook hash mismatch")
print("nbformat",__import__("nbformat").__version__)
nbformat=__import__("nbformat")
nb=nbformat.read(str(book),as_version=4);nbformat.validate(nb)
print("nbformat_valid",nb.nbformat,"cells",len(nb.cells))
code_cells=[(i,c) for i,c in enumerate(nb.cells) if c.cell_type=="code"]
syntax=[]
for i,c in code_cells:
    try: ast.parse(c.source,filename=f"{book}:cell-{i}")
    except SyntaxError as e: syntax.append((i,e.lineno,e.msg))
print("nonempty_code_cells",sum(bool(c.source.strip()) for _,c in code_cells),"ast_errors",syntax)
if syntax: sys.exit("notebook AST errors")
print("first_three_cell_types",[c.cell_type for c in nb.cells[:3]])
print("first_three_code_notebook_indices",[i for i,_ in code_cells[:3]])
# Extract the actual embedded immutable source capsule from the notebook code.
cell=nb.cells[4].source
tree=ast.parse(cell)
payloads=[]
for node in ast.walk(tree):
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="payload" for t in node.targets):
        call=node.value
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=="b64decode":
            payloads.append(base64.b64decode(ast.literal_eval(call.args[0])))
if len(payloads)!=1: sys.exit(f"expected one embedded capsule, got {len(payloads)}")
payload=payloads[0]; capsule_hash=hashlib.sha256(payload).hexdigest()
print("capsule_bytes",len(payload),"capsule_sha256",capsule_hash,"expected",expected_capsule,"match",capsule_hash==expected_capsule)
if capsule_hash!=expected_capsule or m.get("source_sha256")!=expected_capsule: sys.exit("capsule hash mismatch")
expected=m["source_files"]
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    names=z.namelist()
    if len(names)!=len(set(names)): sys.exit("duplicate capsule entries")
    actual={}
    for name in names:
        p=pathlib.PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts: sys.exit("unsafe capsule member:"+name)
        if name.endswith("/"): continue
        actual[name]=hashlib.sha256(z.read(name)).hexdigest()
    missing=sorted(set(expected)-set(actual)); extra=sorted(set(actual)-set(expected))
    bad=sorted((name,expected[name],actual.get(name)) for name in expected if actual.get(name)!=expected[name])
    print("capsule_files",len(actual),"manifest_files",len(expected),"missing",len(missing),"extra",len(extra),"hash_mismatches",len(bad))
    if missing or extra or bad:
        print("missing_sample",missing[:5],"extra_sample",extra[:5],"mismatch_sample",bad[:5]);sys.exit("capsule member hash verification failed")
print("capsule_file_hashes_match",len(expected))
print("kernel_id",meta.get("id"))
print("kernel_private",meta.get("is_private"),"gpu",meta.get("enable_gpu"),"internet",meta.get("enable_internet"))
print("dataset_sources",meta.get("dataset_sources"))
print("kernel_sources",meta.get("kernel_sources"))
if meta.get("id")!="satyapaladuguplus/semantic-decoder-scheduler-20261030" or meta.get("is_private") is not True: sys.exit("kernel identity/privacy mismatch")
if meta.get("code_file")!=book.name: sys.exit("metadata code_file mismatch")
if meta.get("dataset_sources")!=["satyapaladuguplus/semantic-scheduler-paired-inputs-20261102"]: sys.exit("dataset source mismatch")
if meta.get("kernel_sources")!=["satyapaladuguplus/semantic-image-pool-scene-cache-20261030"]: sys.exit("kernel source mismatch")
print("PREFLIGHT_STATIC_PASS")
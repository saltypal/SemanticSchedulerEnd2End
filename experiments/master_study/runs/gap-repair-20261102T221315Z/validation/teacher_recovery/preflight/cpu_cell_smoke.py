"""CPU smoke: run only the first three code cells of the immutable teacher-recovery notebook."""
import hashlib,json,pathlib,sys,time
from types import SimpleNamespace
ROOT=pathlib.Path.cwd()
VAL=ROOT/"experiments/master_study/runs/gap-repair-20261102T221315Z/validation/teacher_recovery/preflight"
BOOK=ROOT/"experiments/master_study/gap_recovery/WSL_Teacher_Recovery.ipynb"
SANDBOX=VAL/"cpu_smoke"
WORKING=SANDBOX/"working"
INPUTS=SANDBOX/"input"
RUN_ROOT=WORKING/"experiments/scheduler_pair/runs"
INPUTS.mkdir(parents=True,exist_ok=True)
WORKING.mkdir(parents=True,exist_ok=True)
source_inputs={
 "Mini_dataset_6000.tar":ROOT/"experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/Mini_dataset_6000.tar",
 "Scheduler_shared_cache_20261102.tar":ROOT/"experiments/master_study/paired_kaggle/inputs/Scheduler_shared_cache_20261102.tar"}
for name,src in source_inputs.items():
 dst=INPUTS/name
 if dst.exists() or dst.is_symlink(): dst.unlink()
 dst.symlink_to(src)
book=json.loads(BOOK.read_text())
cells=[c for c in book["cells"] if c["cell_type"]=="code"]
if len(cells)<3: raise RuntimeError("notebook has fewer than three code cells")
shell=SimpleNamespace(execution_count=0,history_manager=SimpleNamespace(input_hist_raw=[]))
ns={"get_ipython":lambda:shell}
started=time.monotonic()
for i,cell in enumerate(cells[:3]):
 code="".join(cell["source"])
 code=code.replace("Path('/kaggle/working')",f"(Path({str(WORKING)!r}))")
 code=code.replace("Path('/kaggle/input')",f"Path({str(INPUTS)!r})")
 shell.execution_count=i+1
 shell.history_manager.input_hist_raw.append(code)
 print(f"START_CPU_NOTEBOOK_CODE_CELL {i+1}",flush=True)
 exec(compile(code,f"teacher-recovery-cpu-cell-{i+1}","exec"),ns)
 print(f"END_CPU_NOTEBOOK_CODE_CELL {i+1}",flush=True)
run=ns["RUN"]
session=ns["SESSION"]
cfg=json.loads((run/"run_config.json").read_text())
input_manifest=json.loads((run/"input_manifest.json").read_text())
cells_record=json.loads((session/"cells.json").read_text())
if cfg["source_snapshot_sha256"]!="a824d45cb146d850e709a67d29ded1aa2fa8dd3ca1db8248a82a4548fa3291e8": raise RuntimeError("source capsule mismatch")
if len(cfg["source_hashes"])!=182: raise RuntimeError(f"unexpected source hash count: {len(cfg['source_hashes'])}")
if len(cells_record)!=3 or any(r["status"]!="complete" for r in cells_record): raise RuntimeError("not all setup cells completed")
if not (session/"executed_notebook.ipynb").exists(): raise RuntimeError("executed notebook capture missing")
if not (run/"shared/shared_manifest.json").exists(): raise RuntimeError("shared cache extraction missing")
if (RUN_ROOT.parent/"weights").exists(): raise RuntimeError("unexpected weights path")
summary={"notebook":str(BOOK),"executed_code_cells":3,"statuses":[r["status"] for r in cells_record],
 "source_files_verified":len(cfg["source_hashes"]),"source_capsule_sha256":cfg["source_snapshot_sha256"],
 "input_manifest":input_manifest,"run":str(run),"session":str(session),
 "executed_notebook":str(session/"executed_notebook.ipynb"),"gpu_training_executed":False,
 "weights_downloaded":False,"elapsed_seconds":round(time.monotonic()-started,3)}
(SANDBOX/"cpu_cell_smoke.json").write_text(json.dumps(summary,indent=2))
print("CPU_CELL_SMOKE_PASS",json.dumps(summary,sort_keys=True),flush=True)
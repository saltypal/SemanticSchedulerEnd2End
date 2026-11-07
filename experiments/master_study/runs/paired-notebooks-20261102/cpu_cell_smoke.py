"""Execute only session/source/input cells in a fresh CPU process; no fitting."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,tempfile,hashlib
root=Path.cwd();profile=sys.argv[1]
book=json.loads(next((root/"experiments/master_study/paired_kaggle"/profile).glob("*.ipynb")).read_text())
out=root/"experiments/master_study/runs/paired-notebooks-20261102"
out.mkdir(exist_ok=True)
sandbox=out/("cpu_cells_"+profile);sandbox.mkdir(exist_ok=True)
inputs=sandbox/"input";inputs.mkdir(exist_ok=True)
sources={"Mini_dataset_6000.tar":root/"experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/Mini_dataset_6000.tar",
 "Scheduler_shared_cache_20261102.tar":root/"experiments/master_study/paired_kaggle/inputs/Scheduler_shared_cache_20261102.tar"}
for name,path in sources.items():
 target=inputs/name
 if not target.exists():target.symlink_to(path)
shell=SimpleNamespace(execution_count=0,history_manager=SimpleNamespace(input_hist_raw=[]))
ns={"get_ipython":lambda:shell}
cells=[c for c in book["cells"] if c["cell_type"]=="code"]
for i,c in enumerate(cells[:3]):
 code="".join(c["source"]).replace("Path('/kaggle/working')",repr(str(sandbox/"working"))+" and Path("+repr(str(sandbox/"working"))+")")
 code=code.replace("Path('/kaggle/input')","Path("+repr(str(inputs))+")")
 shell.execution_count=i+1;shell.history_manager.input_hist_raw.append(code)
 exec(compile(code,profile+"-cell"+str(i),"exec"),ns)
executed=ns["SESSION"]/"executed_notebook.ipynb"
assert executed.exists()
record={"profile":profile,"executed_code_cells":3,"gpu_training_executed":False,
 "run":str(ns["RUN"]),"session":str(ns["SESSION"]),"executed_notebook":str(executed),
 "source_files":len(json.loads((ns["RUN"]/"run_config.json").read_text())["source_hashes"])}
(out/(profile+"_cpu_cells.json")).write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))

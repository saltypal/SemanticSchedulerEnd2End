"""CPU-only input preflight. Does not infer missing labels or load vision models."""
import argparse,json,sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("boot",ROOT/"experiments/stream_scheduler/runtime_bootstrap.py")
boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.install(ROOT)
from pathlib import Path
from experiments.stream_scheduler.semantic_suite_data import load_pool,cohort_coverage,development_cohorts

def check(archive,output,required_sources):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows,_,_,_,_=load_pool(archive,output/"data")
    coverage=cohort_coverage(rows);train,val=development_cohorts(rows)
    missing=[s for s in required_sources if not coverage.get(s,{}).get("assessable",0)]
    result={"status":"blocked_references" if missing else "ready_scoped_D",
      "coverage":coverage,"missing_reference_sources":missing,"train_images":len(train),"validation_images":len(val),
      "scope":"D class/age only. Mission-evidence study requires separate independent reference manifest."}
    (output/"preflight.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument("--archive",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--require-sources",default="crisismmd,open_images,places365")
    a=p.parse_args();result=check(a.archive,a.output,[s for s in a.require_sources.split(",") if s])
    if result["status"]=="blocked_references":raise SystemExit(2)
if __name__=="__main__":main()

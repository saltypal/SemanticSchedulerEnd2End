"""Merge two downloaded run roots only when source, inputs and evaluator traces match."""
from pathlib import Path
import argparse,csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from Scheduler.training.study.paired_runtime import SUPERVISED,WINDOWS0,WINDOWS1,P_ARMS

def merge(roots,output):
    roots=[Path(p) for p in roots];output=Path(output);output.mkdir(parents=True,exist_ok=True)
    configs=[json.loads((p/"run_config.json").read_text()) for p in roots]
    if {c["profile"] for c in configs}!={"wsl","windows"}:raise ValueError("one run from each profile required")
    if len({c["source_snapshot_sha256"] for c in configs})!=1:raise ValueError("different source snapshots; no pooled comparison")
    inputs=[json.loads((p/"input_manifest.json").read_text()) for p in roots]
    for key in ("archive_sha256","shared_sha256","vectors_sha256","mission_vectors_sha256"):
        if len({i[key] for i in inputs})!=1:raise ValueError("different input identity:"+key)
    traces=[];rows=[];preferences=[];baselines=None
    for root,config in zip(roots,configs):
        for status_path in root.glob("jobs/*/*/status.json"):
            family=status_path.parent.name;job=status_path.parent.parent.name
            baseline_file=status_path.parent/"baseline_summary.json"
            if baseline_file.exists():
                current={k:v["score"] for k,v in json.loads(baseline_file.read_text()).items()}
                if baselines is None:baselines=current
                elif any(abs(current[k]-v)>1e-10 for k,v in baselines.items()):
                    raise ValueError("baseline results differ across supposedly matched traces")
            assigned=(SUPERVISED if job=="supervised" else WINDOWS0 if job=="double_dqn" else WINDOWS1 if job=="ppo_per" else P_ARMS if family=="P" else ("rl_vlm_f","vlm_replay"))
            state=json.loads(status_path.read_text());state=state.get("arms",state)
            tracefile=status_path.parent/"validation_trace_identity.json"
            trace=json.loads(tracefile.read_text())["fingerprints"] if tracefile.exists() else None
            if family in ("D","feedback") and trace is not None:traces.append(trace)
            for key,value in state.items():
                arm,seed=key.rsplit("/",1)
                if arm not in assigned:continue
                row={"profile":config["profile"],"job":job,"method":arm,"seed":seed,"status":value["status"],
                     "role":"warmstart_prerequisite" if config["profile"]=="windows" and arm=="deepsets_ranknet" else "comparison",
                     "delay_cost":None,"teacher_agreement":None,"trace_verified":trace is not None}
                if family in ("D","feedback"):
                    score=value.get("validation_score",value.get("validation"))
                    if score is not None and trace is not None:row["delay_cost"]=-score
                else:row["teacher_agreement"]=value.get("validation",{}).get("silver_agreement")
                (preferences if family=="P" else rows).append(row)
    if traces and any(t!=traces[0] for t in traces):raise ValueError("evaluator observation/outcome/reference traces differ")
    if baselines:
        rows=[{"profile":"shared","job":"baselines","method":k,"seed":"1729","status":"evaluated_development_only",
              "role":"baseline","delay_cost":-v,"teacher_agreement":None,"trace_verified":bool(traces)} for k,v in baselines.items()]+rows
    for filename,data in (("scheduler_comparison.csv",rows),("teacher_preference_comparison.csv",preferences)):
        with (output/filename).open("w",newline="") as f:
            fields=list((data or [{"status":""}])[0]);writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(data)
    (output/"MERGE.json").write_text(json.dumps({"source_match":True,"input_match":True,"trace_match":bool(traces),
        "reference_scope":"development/silver only","delay_rows":len(rows),"preference_rows":len(preferences),
        "note":"rows without trace identity have no comparable delay cost; Windows initialization not independent seed"},indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--wsl",type=Path,required=True);p.add_argument("--windows",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();merge([a.wsl,a.windows],a.output)

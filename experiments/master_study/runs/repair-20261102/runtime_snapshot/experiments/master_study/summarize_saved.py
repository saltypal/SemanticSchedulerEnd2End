"""Offline reporting from saved child results; never starts a learner."""
import argparse,csv,json
from pathlib import Path

def critical_summary(episodes):
    rows=[e["by_reference_tier"]["Critical"] for e in episodes]
    offered=sum(r["offered"] for r in rows)
    follow=sum(r.get("followup_count",0) for r in rows)
    return {"offered":offered,"misses":sum(r["unselected"] for r in rows),
            "restricted_mean_wait":sum((r.get("restricted_mean_wait") or 0)*r.get("followup_count",0) for r in rows)/follow if follow else None}

def priority_guard(candidate,reference):
    if not candidate["offered"] or candidate["offered"]!=reference["offered"]:
        return {"pass":False,"reason":"missing or unmatched critical cohort"}
    ok=(candidate["misses"]<=reference["misses"] and candidate["restricted_mean_wait"] is not None
        and reference["restricted_mean_wait"] is not None and candidate["restricted_mean_wait"]<=reference["restricted_mean_wait"]+1e-12)
    return {"pass":ok,"reason":"engineering critical-wait/miss safeguard, not a statistical final test"}


def summarize(results,output,seed="1729"):
    results=Path(results);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for family in ("D","P","feedback"):
        path=results/family/"status.json"
        if not path.exists():continue
        records=json.loads(path.read_text());records=records.get("arms",records)
        for key,value in records.items():
            arm,trial=key.rsplit("/",1)
            state=value["status"]
            if state=="running":state="interrupted"
            rows.append({"family":family,"method":arm,"seed":trial,"status":state,
                         "score":value.get("validation_score",value.get("validation")),
                         "reason":value.get("reason","")})
    responses=results/"teacher/responses.jsonl";guide="not_retrieved"
    if responses.exists():
        values=[json.loads(x).get("parsed",{}).get("value") for x in responses.read_text().splitlines() if json.loads(x).get("task")=="replay"]
        if values and set(values)=={"neutral"}:guide="inactive_all_neutral"
    with (output/"inventory.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["method","status"]);writer.writeheader();writer.writerows(rows)
    baseline_path=results/"D/baseline_summary.json";comparison=[]
    if baseline_path.exists():
        baselines=json.loads(baseline_path.read_text())
        for name,value in baselines.items():comparison.append({"method":name,"mean_wait_penalty":-value["score"],"scope":"shared_D_validation"})
        for row in rows:
            if row["family"]=="D" and row["seed"]==seed and row["status"]=="evaluated_development_only":
                comparison.append({"method":row["method"],"mean_wait_penalty":-row["score"],"scope":"shared_D_validation"})
    safeguards={}
    if baseline_path.exists() and "spi_fifo" in baselines:
        reference=critical_summary(baselines["spi_fifo"]["episodes"])
        for item in comparison:
            name=item["method"]
            if name in baselines:
                episodes=baselines[name]["episodes"]
            else:
                paths=list((results/"D/models"/name/seed/"validation").glob("episode_*.json"))
                episodes=[json.loads(path.read_text())["metrics"] for path in paths]
            if len(episodes)!=len(baselines["spi_fifo"]["episodes"]):
                safeguards[name]={"pass":False,"reason":"complete matched per-class metrics unavailable"};continue
            safeguards[name]={"metrics":critical_summary(episodes),**priority_guard(critical_summary(episodes),reference)}
    (output/"critical_safeguards.json").write_text(json.dumps(safeguards,indent=2))
    comparison.sort(key=lambda x:x["mean_wait_penalty"])
    with (output/"delay_comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["method","mean_wait_penalty","scope"]);w.writeheader();w.writerows(comparison)
    status={"scope":"development only; fixed first seed, no best-seed selection","seed":seed,
      "complete_methods":sorted({(r["family"]+"/"+r["method"]) for r in rows if r["status"].startswith("evaluated")}),
      "teacher_replay":guide,"rows":len(rows),"independent_final_evaluation":"not_established"}
    (output/"summary.json").write_text(json.dumps(status,indent=2))
    text=["# Saved-run report","",status["scope"],"",
      "Informative VLM guidance: "+guide,"","| Method | D waiting penalty (lower better) |","|---|---:|"]
    text.extend("| "+r["method"]+" | "+format(r["mean_wait_penalty"],".6f")+" |" for r in comparison)
    text.extend(["","No teacher-feedback score is compared against D because historical workloads differ.",
      "See inventory.csv for blocked/interrupted arms. These outputs do not establish final generalization, convergence or semantic correctness."])
    (output/"REPORT.md").write_text("\n".join(text)+"\n")
    return status
def main():
    p=argparse.ArgumentParser();p.add_argument("--results",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--seed",default="1729")
    a=p.parse_args();print(json.dumps(summarize(a.results,a.output,a.seed),indent=2))
if __name__=="__main__":main()

"""CPU development study: offered UE rates/bursts, same traces for all policies."""
import argparse,csv,hashlib,importlib.util,json,platform,sys,time
from dataclasses import asdict
from pathlib import Path
import numpy as np

# Direct-script Kaggle execution starts with this directory on sys.path.
# Install the same isolated canonical namespaces as the training drivers.
ROOT=Path(__file__).resolve().parents[2]
_spec=importlib.util.spec_from_file_location("rate_boot",Path(__file__).with_name("runtime_bootstrap.py"))
_boot=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_boot);_boot.install(ROOT)

from Evaluation.ue_scenarios import StreamConfig,build_stream
from Evaluation.semantic_learning_env import Environment,Rule,PROFILES
from experiments.stream_scheduler.semantic_suite_data import load_pool,load_mission_vectors,split_records

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--vector-extension",type=Path,required=True)
    parser.add_argument("--mission-vectors",type=Path,required=True)
    parser.add_argument("--annotations",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--duration",type=float,default=200.)
    parser.add_argument("--ues",default="12,50,60,100")
    parser.add_argument("--seeds",default="1729,2718,31415")
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    rows,features,vectors,profile,audit=load_pool(args.archive,out/"inputs",args.vector_extension)
    rows=split_records(rows,"train",True)  # Development exposed labels; never final test.
    text=load_mission_vectors(args.mission_vectors)
    if args.annotations:
        annotations=[json.loads(line) for line in args.annotations.read_text().splitlines() if line.strip()]
        modes=("mixed","mission","incident")
        pool_scope="declared manifest; validity remains the dataset owner's responsibility"
    else:
        available=[r for r in rows if r["image_sha256"] in vectors]
        matrix=np.stack([vectors[r["image_sha256"]] for r in available])@text.T
        thresholds=np.quantile(matrix,.75,axis=0)
        annotations=[{"image_id":r["image_id"],"kind":"routine",
                      "missions":[PROFILES[j][0] for j in range(15) if matrix[i,j]>=thresholds[j]],
                      "provenance":"CLIP upper-quartile mission proxy; not independent labels"}
                     for i,r in enumerate(available)]
        modes=("mixed","mission");pool_scope="CLIP mission sampling proxy only; no real incident correspondence"
    source_files=("Evaluation/ue_scenarios.py","Evaluation/semantic_learning_env.py",str(Path(__file__).relative_to(ROOT)))
    manifest={"config":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              "source_hashes":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files},
              "python":sys.version,"platform":platform.platform(),"input_profile":profile,"pool_scope":pool_scope,
              "reference_scope":"development-exposed VLM silver tiers; not independent emergency truth",
              "incident_status":"enabled with manifest" if args.annotations else "unavailable: no supported incident/sequence labels"}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    results=[]
    for seed in map(int,args.seeds.split(",")):
      for ues in map(int,args.ues.split(",")):
       settings=[("fixed_total",value) for value in (.5,.9,1.2)]+[("fixed_per_ue",ues*value) for value in (.01,.02)]
       for design,load in settings:
        for burst in (1.,4.):
         for mode in modes:
          config=StreamConfig(mode=mode,ues=ues,duration=args.duration,load=load,burst_factor=burst,seed=seed)
          episode=build_stream(rows,features,vectors,text,config,annotations)
          ident=f"{seed}-{ues}-{design}-{load:.4f}-{burst}-{mode}"
          traces=out/"traces";traces.mkdir(exist_ok=True)
          (traces/(ident+".json")).write_text(json.dumps(episode.metadata))
          for policy_name in ("fifo","round_robin","spi_fifo","spi_novelty"):
            env=Environment(episode);policy=Rule(policy_name,seed);peak=0;timings=[]
            while not env.done:
                state=env.snapshot();peak=max(peak,len(state.ready))
                start=time.perf_counter_ns();token=policy.select(state);timings.append((time.perf_counter_ns()-start)/1e6)
                env.step(token)
            metrics=env.metrics()
            selected={e["token"]:e["dispatch"] for e in env.events}
            waiting_sum=sum(max(0,min(selected.get(m.token,episode.horizon),episode.horizon)-m.arrival) for m in episode.messages)
            record={"trace":ident,"policy":policy_name,"rate_design":design,"mode":mode,"ues":ues,"target_load":load,
                    "burst_factor":burst,"seed":seed,"offered":len(episode.messages),"realized_rate":episode.metadata["realized_rate"],
                    "realized_service_load":episode.metadata["offered_service_load"],"peak_ready":peak,
                    "mean_queue_length":waiting_sum/episode.horizon,
                    "restricted_mean_wait":waiting_sum/len(episode.messages) if episode.messages else None,
                    "undispatched_fraction":metrics["queued_at_horizon"]/len(episode.messages) if episode.messages else None,
                    "selection_p50_ms":float(np.median(timings)) if timings else None,"metrics":metrics}
            results.append(record)
          (out/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False))
          print("STREAM_STUDY",ident,"offered",len(episode.messages),"rows",len(results),flush=True)
    flat=[{k:v for k,v in row.items() if k!="metrics"} for row in results]
    with (out/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    (out/"REPORT.md").write_text(
        "# UE rate development study\n\n"+str(len(results))+" paired policy rows.\n\n"+
        "Rates use synthetic time units; service time is1unit. Fixed-total load separates UE count from aggregate demand. "+
        "Fixed-per-UE rate allows demand to grow with population. Burst intensity is normalized so its full-window expected rate is unchanged before UE availability effects. "+
        "Record both intended and realized load; joining/pausing UEs reduce achieved traffic.\n\n"+
        pool_scope+". Incident mode is unavailable without a supported manifest. Do not claim natural temporal understanding from mixed images or CLIP sampling proxies. "+
        "Delay includes censored waiting to the fixed horizon; class-specific restricted waits and defined counts remain in results.json. No final test or model training occurred.\n")
    print("COMPLETE",out,flush=True)
if __name__=="__main__":main()

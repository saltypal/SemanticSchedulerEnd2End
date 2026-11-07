"""Pinned metadata-only diversity selection. Native labels never become SPI."""
import json,hashlib,urllib.request
from collections import defaultdict,Counter
from pathlib import Path
REVISION="10a5626ba112ab9c50369c473a2062bcf205d708"

def select(output,count=1000,metadata_root=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    records={};conflicts=[]
    for task in ("informative","humanitarian","damage"):
        for split in ("train","dev","test"):
            url=f"https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/{REVISION}/{task}/{split}.json"
            cached=Path(metadata_root)/f'{task}_{split}.json' if metadata_root else None
            if cached is not None and cached.exists():raw=cached.read_bytes()
            else:
                with urllib.request.urlopen(url,timeout=60) as response:raw=response.read()
            (out/f"{task}_{split}.json").write_bytes(raw)
            for source in json.loads(raw):
                key=source["image_path"]
                row=records.setdefault(key,{k:v for k,v in source.items() if k!="label"})
                row.setdefault("native_annotations",{}).setdefault(task,[]).append({"split":split,"label":source["label"]})
    buckets=defaultdict(list)
    for path,row in records.items():
        # Retain annotation conflicts rather than inventing one agreed label.
        labels={k:tuple(sorted({v["label"] for v in values})) for k,values in row["native_annotations"].items()}
        if any(len(v)>1 for v in labels.values()):conflicts.append(path)
        key=(row["event_name"],labels.get("humanitarian",("unknown",)),labels.get("damage",("unknown",)),labels.get("informative",("unknown",)))
        buckets[key].append(row)
    for values in buckets.values():values.sort(key=lambda r:hashlib.sha256(("seed29:"+r["image_path"]).encode()).hexdigest())
    selected=[];keys=sorted(buckets,key=str)
    while len(selected)<count:
        progress=False
        for key in keys:
            if buckets[key]:selected.append(buckets[key].pop());progress=True
            if len(selected)==count:break
        if not progress:raise ValueError("insufficient source images")
    # Parent tweet identity stays one role; exact/near duplicates need image-byte audit next.
    rank={"train":0,"dev":1,"test":2};roles={}
    for row in selected:
        native=max((v["split"] for values in row["native_annotations"].values() for v in values),key=rank.get)
        group=str(row["tweet_id"]);roles[group]=max((roles.get(group,"train"),native),key=rank.get)
    for row in selected:row["group"]="tweet:"+str(row["tweet_id"]);row["split"]=roles[str(row["tweet_id"])];row["source_revision"]=REVISION
    (out/"selected_1000.json").write_text(json.dumps(selected,indent=2),encoding="utf8")
    summary={"count":len(selected),"events":dict(Counter(r["event_name"] for r in selected)),"splits":dict(Counter(r["split"] for r in selected)),"native_strata":len(keys),"conflicting_source_images":len(conflicts),"selection":"seed29 deterministic round-robin event x native humanitarian x damage x informativeness","SPI_labels_generated":False,"images_downloaded":0,"text_policy":"preserved separately; image-only SPI teacher does not receive tweet text or annotations","limitations":"enriched sample, not natural prevalence; screenshots not guaranteed by annotations; exact/near image duplicate checks pending"}
    (out/"selection_report.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args();select(a.output)

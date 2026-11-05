"""Finish isolated GPU format/dependency checks; never train/promote from these replies."""
import argparse,hashlib,importlib.util,json,os,re,subprocess,sys,tarfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location("boot",Path(__file__).with_name("runtime_bootstrap.py"))
boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.install(ROOT)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--seconds",type=float,default=720)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    if torch.cuda.device_count()<2:raise RuntimeError("two-device hardware-check profile required")
    ledger={}
    env=dict(os.environ);env["PYTHONPATH"]=str(ROOT)+os.pathsep+str(ROOT/"src")
    result=subprocess.run([sys.executable,"-m","pytest","-q",str(ROOT/"Test/tests/test_channel_mapping.py")],
                          env=env,capture_output=True,text=True)
    (args.output/"cuda_channel_tests.log").write_text(result.stdout+result.stderr)
    ledger["cuda_channel_regression"]={"status":"passed" if result.returncode==0 else "failed","exit":result.returncode}
    (args.output/"validation.json").write_text(json.dumps(ledger,indent=2))
    try:
        requests=[json.loads(line) for line in (args.inputs/"technical_requests.jsonl").read_text().splitlines()]
        prior=json.loads((args.inputs/"prior_qualification.json").read_text())
        if not prior["qualified"]:raise ValueError("previous control qualification did not pass")
        # Use exactly the v5 causal image contexts, rebuilding only image paths from hashes.
        images=args.output/"images";images.mkdir()
        with tarfile.open(args.archive) as archive:
            index=json.load(archive.extractfile("Mini_dataset/portable_image_index.json"))
            by_hash={row["image_sha256"]:row["relative_path"] for row in index}
            for request in requests:
                resolved=[]
                for old in request["images"]:
                    digest=Path(old).stem
                    if not re.fullmatch("[0-9a-f]{64}",digest) or digest not in by_hash:
                        raise ValueError("technical image hash absent from pinned source")
                    path=images/(digest+".jpg")
                    if not path.exists():
                        payload=archive.extractfile("Mini_dataset/"+by_hash[digest]).read()
                        if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError("technical image hash mismatch")
                        path.write_bytes(payload)
                    resolved.append(str(path.resolve()))
                request["images"]=resolved
        query=args.output/"technical_requests.jsonl"
        query.write_text("\n".join(map(json.dumps,requests))+"\n")
        from Scheduler.training.stream_suite.qwen_teacher import run,MODEL_ID
        run(query,args.output/"teacher","cc594898137f460bfe9f0759e9844b3ce807cfb5",
            device="cuda:1",max_calls=len(requests),deadline_seconds=args.seconds)
        answers=[json.loads(line) for line in (args.output/"teacher/responses.jsonl").read_text().splitlines()]
        previous=json.loads((args.inputs/"prior_provenance.json").read_text())
        current=json.loads((args.output/"teacher/provenance.json").read_text())
        if any(previous[key]!=current[key] for key in ("model_id","revision","quantization","weights_sha256","output_version")):
            raise ValueError("teacher identity changed relative to qualified v5 model")
        ledger["teacher_formats"]={"status":"passed" if len(answers)==len(requests) and all(row["status"]=="parsed" for row in answers) else "failed_or_partial",
                                  "model":MODEL_ID,"requested":len(requests),"responses":len(answers),
                                  "tasks":[row["task"] for row in answers],
                                  "weights_match_v5":True,"prior_controls_qualified":True,
                                  "no_training_or_label_promotion":True,
                                  "scope":"isolated structured-output hardware checks; prior narrow controls are not emergency-priority truth"}
    except Exception:
        ledger["teacher_formats"]={"status":"failed","traceback":traceback.format_exc()}
    (args.output/"validation.json").write_text(json.dumps(ledger,indent=2))
    print("TARGETED_GPU_VALIDATION",json.dumps(ledger),flush=True)
    return int(any(row["status"]!="passed" for row in ledger.values()))


if __name__=="__main__":raise SystemExit(main())

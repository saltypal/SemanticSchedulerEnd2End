"""Offline Qwen image teacher. Raw answers are weak labels, never reference truth."""
import argparse,hashlib,json,time,re
from pathlib import Path

MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
INFERENCE_PROTOCOL={"version":"labeled_teacher_panels_v1","use_fast":False,
                    "standard_pixels":256*28*28,"panel_max_pixels":2048*28*28,
                    "decoding":"greedy_max512_nf4"}

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.stream_scheduler.teacher_output import (
    OUTPUT_VERSION, answer_schema, parse_answer, prefix_factory)

def run(request_path,output,revision,device="cuda:0",max_calls=216,deadline_seconds=None):
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,BitsAndBytesConfig
    from PIL import Image,ImageOps
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not re.fullmatch("[0-9a-f]{40}",revision):raise ValueError("pin an immutable official model revision")
    if deadline_seconds is not None and not 0 < deadline_seconds < float("inf"):raise ValueError("positive finite teacher time budget required")
    if max_calls>216 or max_calls<1:raise ValueError("invalid total-call ceiling")
    requests=[json.loads(l) for l in Path(request_path).read_text().splitlines() if l.strip()]
    if len({r["id"] for r in requests})!=len(requests):raise ValueError("duplicate query IDs")
    renderer_hash=hashlib.sha256(Path(__file__).with_name("teacher_panels.py").read_bytes()).hexdigest()
    raw=output/"responses.jsonl";done={}
    if raw.exists():
        for line in raw.read_text().splitlines():
            r=json.loads(line);done[r["id"]]=r
    if done:
        prior=json.loads((output/"provenance.json").read_text())
        if prior.get("inference_protocol")!=INFERENCE_PROTOCOL:raise ValueError("teacher visual protocol changed; use a fresh output directory")
        if prior.get("renderer_sha256")!=renderer_hash:raise ValueError("teacher renderer changed; use a fresh output directory")
        if prior.get("output_version") != OUTPUT_VERSION:raise ValueError("teacher cache uses a different output grammar; preserve it and use a new run")
        if prior["model_id"]!=MODEL_ID or prior["revision"]!=revision or prior["quantization"]!="NF4_double_FP16":raise ValueError("incompatible teacher response cache")
        by_id={r["id"]:r for r in requests}
        for ident,answer in done.items():
            if ident not in by_id:raise ValueError("cached query absent from current request set")
            request=by_id[ident]
            current=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
            hashes=[hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in request["images"]]
            if answer["request_sha256"]!=current or answer["image_sha256"]!=hashes:raise ValueError("teacher request/image cache identity mismatch")
    if sum(q["task"]=="control" and q["id"] in done for q in requests)==32:
        from experiments.stream_scheduler.teacher_pipeline import qualify
        gate=qualify(request_path,raw)
        if not gate["qualified"]:raise RuntimeError("Cached controls failed qualification; refuse further collection")
    if done:
        from experiments.stream_scheduler.teacher_pipeline import preference_reliability
        prior_quality=preference_reliability(request_path,raw,complete_only=True)
        if prior_quality["contexts"]>=8 and not prior_quality["qualified"]:
            raise RuntimeError("Cached scheduling pilot failed; preserve rejected labels and use a new protocol/run")
    if len(done)>=max_calls or len(done)==len(requests):return
    if not torch.cuda.is_available():raise RuntimeError("teacher requires verified CUDA")
    started=time.monotonic()
    from lmformatenforcer import JsonSchemaParser  # Required before loading weights.
    processor=AutoProcessor.from_pretrained(MODEL_ID,revision=revision,use_fast=False,min_pixels=256*28*28,max_pixels=256*28*28)
    panel_processor=AutoProcessor.from_pretrained(MODEL_ID,revision=revision,use_fast=False,min_pixels=256*28*28,max_pixels=2048*28*28)
    grammar_start=time.monotonic();constraint=prefix_factory(processor.tokenizer);grammar_setup=time.monotonic()-grammar_start
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID,revision=revision,quantization_config=quant,torch_dtype=torch.float16,device_map={"":device},attn_implementation="sdpa").eval()
    if any(str(p.device)!=device for p in model.parameters()):raise RuntimeError("unexpected teacher placement")
    from bitsandbytes.nn import Linear4bit
    from huggingface_hub import snapshot_download
    from experiments.stream_scheduler.semantic_suite_data import file_hash
    quantized=sum(isinstance(m,Linear4bit) for m in model.modules())
    if not quantized:raise RuntimeError("NF4 modules were not activated")
    checkpoint=Path(snapshot_download(MODEL_ID,revision=revision,local_files_only=True))
    provenance={"inference_protocol":INFERENCE_PROTOCOL,"renderer_sha256":renderer_hash,"grammar_tokenizer_setup_seconds":grammar_setup,"grammar_implementation":"shared_static_tokenizer_fresh_parser","output_version":OUTPUT_VERSION,"constrained_generation":True,"weights_sha256":{p.name:file_hash(p) for p in checkpoint.glob("*.safetensors")},"linear4bit_modules":quantized,"model_footprint_bytes":model.get_memory_footprint(),"model_id":MODEL_ID,"revision":revision,"device":device,"quantization":"NF4_double_FP16","torch":torch.__version__,"weak_supervision":True}
    (output/"provenance.json").write_text(json.dumps(provenance,indent=2))
    for r in requests:
        if r["id"] in done:continue
        if len(done)>=max_calls or (deadline_seconds is not None and time.monotonic()-started>=deadline_seconds):break
        paths=[Path(x) for x in r["images"]]
        if not 1<=len(paths)<=10:raise ValueError("teacher image count outside contract")
        images=[ImageOps.exif_transpose(Image.open(x)).convert("RGB") for x in paths]
        hashes=[hashlib.sha256(x.read_bytes()).hexdigest() for x in paths]
        content=[]
        for i in range(len(images)):
            content.extend([{"type":"text","text":r["roles"][i]},{"type":"image"}])
        content.append({"type":"text","text":r["prompt"] + "\nOutput exactly one JSON object conforming to this schema: " + json.dumps(answer_schema(r["task"]))})
        messages=[{"role":"system","content":"Judge only visible evidence for the stated mission. Do not infer unseen danger or trust device identity. Return one JSON object, allow tie and uncertain. Evidence must be concise; unknowns must be a list."},{"role":"user","content":content}]
        if r.get("visual_layout"):
            if r["visual_layout"]!=INFERENCE_PROTOCOL["version"] or r.get("pixel_budget")!=INFERENCE_PROTOCOL["panel_max_pixels"]:
                raise ValueError("unsupported labeled-panel request protocol")
            active_processor=panel_processor
        else:
            active_processor=processor
        text=active_processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=active_processor(text=[text],images=images,return_tensors="pt",padding=True).to(device)
        t=time.monotonic();record={"id":r["id"],"task":r["task"],"request_sha256":hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest(),"image_sha256":hashes}
        try:
            with torch.inference_mode():result=model.generate(**inputs,max_new_tokens=512,do_sample=False,
                prefix_allowed_tokens_fn=constraint(r["task"]))
            torch.cuda.synchronize(device)
            answer=processor.batch_decode(result[:,inputs["input_ids"].shape[1]:],skip_special_tokens=True)[0]
            record["raw"]=answer
            record["output_version"]=OUTPUT_VERSION
            record["generated_tokens"]=int(result.shape[1]-inputs["input_ids"].shape[1])
            try:record["parsed"]=parse_answer(answer,r["task"]);record["status"]="parsed"
            except Exception as e:record["status"]="invalid";record["error"]=str(e)
        except Exception as e:record["status"]="failed";record["error"]=repr(e)
        record["seconds"]=time.monotonic()-t
        with raw.open("a") as f:f.write(json.dumps(record)+chr(10));f.flush()
        done[r["id"]]=record;print("TEACHER",r["id"],record["status"],record["seconds"],flush=True)
        for image in images:image.close()
        if r["task"]=="control" and sum(q["task"]=="control" and q["id"] in done for q in requests)==32:
            from experiments.stream_scheduler.teacher_pipeline import qualify
            gate=qualify(request_path,raw);(output/"qualification.json").write_text(json.dumps(gate,indent=2))
            if not gate["qualified"]:raise RuntimeError("Teacher qualification failed; do not collect training labels: "+json.dumps(gate))
        if r["task"]=="candidate":
            from experiments.stream_scheduler.teacher_pipeline import preference_reliability
            quality=preference_reliability(request_path,raw,complete_only=True)
            (output/"preference_reliability.json").write_text(json.dumps(quality,indent=2))
            if quality["contexts"]>=8 and not quality["qualified"]:
                raise RuntimeError("Scheduling preference pilot failed; stop collection: "+json.dumps(quality))

if __name__=="__main__":
    import importlib.util
    spec=importlib.util.spec_from_file_location("boot",Path(__file__).with_name("runtime_bootstrap.py"));boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.install(Path(__file__).resolve().parents[2])
    p=argparse.ArgumentParser();p.add_argument("--requests",required=True);p.add_argument("--output",required=True);p.add_argument("--revision",required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--max-calls",type=int,default=216);p.add_argument("--deadline-seconds",type=float,default=None)
    a=p.parse_args();run(a.requests,a.output,a.revision,a.device,a.max_calls,a.deadline_seconds)

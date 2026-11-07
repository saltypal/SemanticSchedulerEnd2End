"""Fill only missing image embeddings with the original pinned CLIP producer."""
import argparse,importlib.util,io,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sp=importlib.util.spec_from_file_location("boot",Path(__file__).with_name("runtime_bootstrap.py"));b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b);b.install(ROOT)
import numpy as np
from Scheduler.training.stream_suite.semantic_suite_data import load_pool,CLIP_ID,CLIP_REV,file_hash

def main():
    import torch
    from PIL import Image,ImageOps
    from transformers import CLIPModel,AutoProcessor
    from huggingface_hub import snapshot_download
    # Reuse only the frozen producer modules; the package initializer imports unrelated legacy owners.
    import sys,types
    if "parameters" not in sys.modules:
        package=types.ModuleType("parameters");package.__path__=[str(ROOT/"parameters")];sys.modules["parameters"]=package
    from parameters.vision import FrozenVisionProvider
    p=argparse.ArgumentParser();p.add_argument("--archive",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    rows,features,vectors,profile,audit=load_pool(a.archive,a.output/"data")
    checkpoint=Path(snapshot_download(CLIP_ID,revision=CLIP_REV,allow_patterns=["*.json","*.txt","pytorch_model.bin"]))
    for name,sha in profile["semantic_producer"]["weights_sha256"].items():
        if file_hash(checkpoint/name)!=sha:raise ValueError("pinned visual checkpoint mismatch")
    model=CLIPModel.from_pretrained(checkpoint,local_files_only=True,use_safetensors=False).to(a.device).eval()
    processor=AutoProcessor.from_pretrained(checkpoint,local_files_only=True)
    missing={r["image_sha256"]:r["image_id"] for r in rows if r["image_sha256"] not in vectors}
    tar=tarfile.open(a.archive);index={r["image_id"]:r for r in json.load(tar.extractfile("Mini_dataset/portable_image_index.json"))}
    result={};items=list(missing.items())
    for offset in range(0,len(items),32):
        images=[];batch=items[offset:offset+32]
        for sha,ident in batch:
            data=tar.extractfile("Mini_dataset/"+index[ident]["relative_path"]).read()
            import hashlib
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError("source image hash mismatch")
            image=ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            images.append(FrozenVisionProvider._pad(image,224))
        inputs=processor(images=images,do_resize=False,do_center_crop=False,return_tensors="pt").to(a.device)
        with torch.inference_mode():value=model.get_image_features(**inputs);value=value/value.norm(dim=-1,keepdim=True)
        for (sha,_),v in zip(batch,value.cpu().numpy()):result[sha]=v.astype(np.float32)
        for image in images:image.close()
        print("MISSING_CLIP",offset+len(batch),len(items),flush=True)
    tar.close();np.savez(a.output/"vectors.npz",**result)
    (a.output/"identity.json").write_text(json.dumps({"archive_sha256":file_hash(a.archive),"encoder":profile["semantic_producer"],"added":len(result),"retained_cached":len(vectors),"vectors_sha256":file_hash(a.output/"vectors.npz")},indent=2))
if __name__=="__main__":main()

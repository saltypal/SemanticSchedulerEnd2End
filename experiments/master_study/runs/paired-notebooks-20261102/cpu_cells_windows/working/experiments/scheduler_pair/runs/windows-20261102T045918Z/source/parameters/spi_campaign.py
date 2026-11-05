"""Bounded development input packaging and offline teacher -> student execution.

Reuse real cached image features. Source annotations are never teacher inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import io
import numpy as np

from .artifacts import write_json, Run
from .contracts import digest
from .extractor import VisualEvidence, embedding_signature
from .spi_labels import load_rubric, label_image
from .spi import TrainingConfig, train


def prepare_cached_cohort(source, output, checkpoint, max_image_bytes=250_000_000):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError("use a new immutable input directory")
    records=json.loads((source/"data/records.json").read_text(encoding="utf8"))
    split=json.loads((source/"metrics/clip/split.json").read_text(encoding="utf8"))
    training,validation=set(split["train_groups"]),set(split["validation_groups"])
    if training & validation:raise ValueError("cached development groups overlap")
    output.mkdir(parents=True)
    from PIL import Image
    from .minimal_vision import MinimalVisionProvider,prepare_image,INPUT_POLICY
    from .contracts import MissionSpec
    mission=MissionSpec("spi_reference","Review visible images.",(("visible","an image"),))
    provider=MinimalVisionProvider(checkpoint,{mission.mission_id:mission})
    rows=[];embeddings=[];total=0;original_bytes=0;hashes=set();signatures=set()
    try:
        with zipfile.ZipFile(output/"images.zip","w",zipfile.ZIP_STORED) as archive:
            for start in range(0,len(records),8):
                pictures=[]
                for r in records[start:start+8]:
                    path=Path(r["image_path"])
                    if not path.is_absolute():path=source/path
                    original_bytes+=path.stat().st_size
                    with Image.open(path) as original:picture=prepare_image(original)
                    buffer=io.BytesIO();picture.save(buffer,format="PNG");raw=buffer.getvalue();total+=len(raw)
                    if total>max_image_bytes:raise RuntimeError("bounded image-package cap exceeded")
                    sha=hashlib.sha256(raw).hexdigest()
                    if sha in hashes:raise ValueError("derived exact duplicates must be collapsed before packaging")
                    hashes.add(sha)
                    role="train" if r["group"] in training else "validation" if r["group"] in validation else None
                    if role is None:raise ValueError("record lacks a declared development split")
                    rows.append({"image_id":r["image_id"],"image_sha256":sha,"image_path":"images/"+sha+".png",
                        "group":r["group"],"split":role,"source":r["source"],"development_exposed":True})
                    archive.writestr(rows[-1]["image_path"],raw);pictures.append(picture)
                evidence=provider.extract_batch(pictures,[mission]*len(pictures))
                embeddings.extend(e.embedding for e in evidence);signatures.update(embedding_signature(e) for e in evidence)
                print("Prepared SPI inputs/features",len(rows),"/",len(records),"bytes",total,flush=True)
    finally:provider.close()
    if len(signatures)!=1:raise ValueError("mixed encoder/preprocessing signatures")
    np.savez_compressed(output/"features.npz",image_ids=np.array([r["image_id"] for r in rows]),
        image_sha256=np.array([r["image_sha256"] for r in rows]),embeddings=np.asarray(embeddings,np.float32),
        feature_signature=np.array(next(iter(signatures))))
    write_json(output/"panel.json",rows)
    write_json(output/"input_manifest.json",{"schema":"spi_inputs_v1","source":str(source.resolve()),
        "images":len(rows),"source_image_bytes":original_bytes,"packaged_image_bytes":total,
        "input_policy":INPUT_POLICY,"development_exposed":True,
        "native_labels_included":False,"feature_signature":next(iter(signatures)),
        "files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}})
    print("Prepared",len(rows),"real images; native annotations excluded; image bytes",total,flush=True)
    return output


def verify_inputs(input_root):
    """Kaggle expands uploaded ZIPs. Verify exact contents in either layout."""
    input_root=Path(input_root).resolve()
    rows=json.loads((input_root/"panel.json").read_text(encoding="utf8"))
    manifest=json.loads((input_root/"input_manifest.json").read_text(encoding="utf8"))
    for name,expected in manifest["files"].items():
        if name=="images.zip":continue
        path=(input_root/name).resolve()
        if not path.is_relative_to(input_root) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError("input manifest checksum mismatch: "+name)
    archive=input_root/"images.zip"
    if archive.exists():
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest["files"]["images.zip"]:
            raise ValueError("input archive checksum mismatch")
        with zipfile.ZipFile(archive) as zipped:
            for row in rows:
                if hashlib.sha256(zipped.read(row["image_path"])).hexdigest()!=row["image_sha256"]:
                    raise ValueError("archived image checksum mismatch")
        return rows,{"layout":"zip","images_verified":len(rows),"archive_hash_verified":True}
    # Observed Kaggle transformation: images.zip -> images/<ZIP member path>.
    for row in rows:
        path=(input_root/"images"/row["image_path"]).resolve()
        if not path.is_relative_to(input_root) or hashlib.sha256(path.read_bytes()).hexdigest()!=row["image_sha256"]:
            raise ValueError("expanded image checksum mismatch")
    return rows,{"layout":"kaggle_expanded_zip","images_verified":len(rows),
                 "archive_hash_verified":False,"reason":"Kaggle expanded archive; exact member hashes verified"}


def execute(input_root,output,reviewer,seconds=3300):
    input_root=Path(input_root);rubric=load_rubric()
    run=Run(output,{"seconds":seconds,"purpose":"offline_SPI_teacher_labelling_and_student_fit",
                    "rubric_sha256":digest(rubric),"enable_thinking":False,"fresh_final_evaluation":False})
    labels=[]
    try:
        with run.stage("input_preflight"):
            rows,receipt=verify_inputs(input_root)
            write_json(run.path/"data/input_verification.json",receipt)
            if receipt["layout"]=="zip":
                unpacked=run.path/"data/images";unpacked.mkdir()
                with zipfile.ZipFile(input_root/"images.zip") as archive:
                    if any(not (unpacked/name).resolve().is_relative_to(unpacked.resolve()) for name in archive.namelist()):
                        raise ValueError("unsafe image archive path")
                    archive.extractall(unpacked)
            else:unpacked=input_root/"images"
        with run.stage("offline_teacher"):
            for index,row in enumerate(rows):
                run.budget();path=unpacked/row["image_path"]
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row["image_sha256"]:raise ValueError("image checksum mismatch")
                label=label_image(reviewer,str(path),row["image_id"],run.path,rubric)
                labels.append({**row,**label})
                write_json(run.path/"data/spi_labels.json",labels)
                if index==0 or (index+1)%10==0:
                    print("SPI teacher",index+1,"/",len(rows),"valid",sum(r["available"] for r in labels),flush=True)
                if index==9 and sum(r["execution"].get("status")=="complete" for r in labels)<8:
                    raise RuntimeError("teacher technical preflight failed: fewer than 8/10 parseable responses; preserve raw evidence")
        reviewer.close()
        with run.stage("fit_student_cpu"):
            _,report=train(input_root/"features.npz",run.path/"data/spi_labels.json",run.path/"student",
                           TrainingConfig(seconds=300))
            write_json(run.path/"metrics/student_validation.json",report)
        write_json(run.path/"status.json",{"status":"trained","labelled_images":len(labels),
            "available_labels":sum(r["available"] for r in labels),"claim":"model-assisted; development data; not human urgency validation"})
    except Exception as error:
        write_json(run.path/"status.json",{"status":"incomplete","reason":str(error),"labelled_images":len(labels)})
        raise
    finally:
        reviewer.close()
        files=[p for p in run.path.rglob('*') if p.is_file() and
               not p.is_relative_to(run.path/'data/images') and p.name not in ('spi_recovery.zip','checksums.json')]
        write_json(run.path/'checksums.json',{str(p.relative_to(run.path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        with zipfile.ZipFile(run.path/'spi_recovery.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for p in files+[run.path/'checksums.json']:archive.write(p,str(p.relative_to(run.path)))
    return run.path


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--cached-source",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--checkpoint",required=True)
    args=parser.parse_args();prepare_cached_cohort(args.cached_source,args.output,args.checkpoint)

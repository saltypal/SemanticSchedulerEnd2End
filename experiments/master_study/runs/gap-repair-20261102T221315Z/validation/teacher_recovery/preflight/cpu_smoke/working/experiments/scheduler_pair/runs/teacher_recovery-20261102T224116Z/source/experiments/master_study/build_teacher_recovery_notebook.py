"""Build the targeted teacher recovery notebook; does not submit or alter paired notebooks."""
import hashlib,io,json,zipfile
from pathlib import Path
from experiments.master_study import build_paired_notebooks as base
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/"experiments/master_study/gap_recovery"

def build():
    original,hashes=base.make_snapshot()
    extras=("experiments/master_study/teacher_recovery.py",
            "experiments/stream_scheduler/teacher_panels.py",
            "experiments/master_study/tests/test_gap_recovery.py",
            "experiments/stream_scheduler/tests/test_teacher_and_supervisor.py",
            "experiments/master_study/build_teacher_recovery_notebook.py")
    existing=zipfile.ZipFile(io.BytesIO(original));files={name:existing.read(name) for name in existing.namelist()}
    for name in extras:files[name]=(ROOT/name).read_bytes()
    packed=io.BytesIO()
    with zipfile.ZipFile(packed,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            hashes[name]=hashlib.sha256(data).hexdigest()
            info=zipfile.ZipInfo(name,(2026,11,3,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,data)
    blob=packed.getvalue()
    book=base.build("windows",blob,hashes,persist=False)
    book["metadata"]["scheduler_profile"]="teacher_recovery"
    book["cells"][0]=base.markdown("""# WSL — targeted teacher recovery
Launch under satyapaladuguplus, existing semantic-decoder-scheduler-20261030 notebook identity.
This recovery prepares fresh labeled candidate/prefix/replay panels, qualifies the same pinned Qwen2.5-VL-7B NF4,
and runs ONLY the seven preference students and two VLM-feedback RL methods if qualified.
The ten D learners and four annotation-reward RL models are preserved, not retrained.
The complete UE-rate gap was independently rerun on CPU/WSL:320rows,80traces, no accounting errors.
SQ-GAN v12 six-payload integration already passed; no codec or encoder inference is repeated here.
Use the existing private image-pool output plus shared-cache dataset; no SQ-GAN checkpoint input is needed.
One seed1729;50epochs max/patience8;20,000 actions per feedback arm.
Controls,80%swap-consistency and16/8strict support/reward-validation gates are unchanged.
VisionPrefer is an explicitly adapted partial-common-aspect mean (>=2known dimensions including mission usefulness);
unknown dimensions are not zero-filled. Query selection after the fixed8-context repair pilot uses permitted feature contrasts.
Teacher labels and held-out preference agreement remain silver and contrast-conditioned.
Training stops at10.5hours; finalization target11.5hours before the12hour platform kill. Setup/model loading count.
""")
    for cell in book["cells"]:
        text="".join(cell["source"])
        if cell["cell_type"]=="code":
            text=text.replace("PROFILE='windows'","PROFILE='teacher_recovery'")
            if "05_contracts_and_reference_scope" in text:
                text=text.replace("str(SOURCE/'experiments/master_study/tests/test_controller.py')]",
                    "str(SOURCE/'experiments/master_study/tests/test_controller.py'),str(SOURCE/'experiments/master_study/tests/test_gap_recovery.py'),str(SOURCE/'experiments/stream_scheduler/tests/test_teacher_and_supervisor.py')]")
            if "07_training_and_conditional_labels" in text:
                text=text.replace("from experiments.master_study.paired_runtime import PairRun","from experiments.master_study.teacher_recovery import TeacherRecovery")
                text=text.replace("runner=PairRun(config)","runner=TeacherRecovery(config)")
            if "08_tables_figures_and_recovery" in text:
                text += "\nfor name in ('qualification.json','preference_reliability.json'):\n    path=RUN/'teacher'/name\n    if path.exists():print(name,json.loads(path.read_text()),flush=True)\n"
            compile(text,"recovery-cell","exec");cell["source"]=text.splitlines(keepends=True)
        elif text.startswith("## 6. Codec boundary"):
            cell["source"]=base.markdown("## 6. Previously verified codec boundary\nSQ-GAN six real payloads passed in WSL version12. This targeted recovery performs no codec/model retraining.")["source"]
        elif text.startswith("## 7. Assigned training"):
            cell["source"]=base.markdown("## 7. Targeted recovery\nQwen qualifies first. Then seven P students use GPU0 while the two feedback RL arms use GPU1. No D/annotation-reward learners run here.")["source"]
    for i,cell in enumerate(book["cells"]):cell["id"]="recovery-"+str(i)
    DEST.mkdir(exist_ok=True)
    path=DEST/"WSL_Teacher_Recovery.ipynb";path.write_text(json.dumps(book,indent=1))
    previous=json.loads((ROOT/"experiments/master_study/paired_kaggle/wsl/kernel-metadata.json").read_text())
    previous.update({"id":"satyapaladuguplus/semantic-decoder-scheduler-20261030",
                     "title":"Semantic Decoder Scheduler 20261030","code_file":path.name,
                     "dataset_sources":["satyapaladuguplus/semantic-scheduler-paired-inputs-20261102"],
                     "kernel_sources":["satyapaladuguplus/semantic-image-pool-scene-cache-20261030"]})
    (DEST/"kernel-metadata.json").write_text(json.dumps(previous,indent=2))
    manifest={"notebook":str(path.relative_to(ROOT)),"notebook_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_sha256":hashlib.sha256(blob).hexdigest(),"source_files":hashes,"submitted":False,
              "new_label_protocol":"labeled_teacher_panels_v1",
              "purpose":"nine blocked teacher arms only; completed14models unchanged",
              "rate_results":"experiments/stream_scheduler/runs/gap-repair-20261102T221315Z/rate_cpu_full"}
    (DEST/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!="source_files"},indent=2))
if __name__=="__main__":build()

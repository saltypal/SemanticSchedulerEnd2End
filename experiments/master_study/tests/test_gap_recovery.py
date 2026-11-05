"""Regression checks for real repair failures, teacher binding and recovery scope."""
import json,subprocess,sys,time
from pathlib import Path
import pytest
from PIL import Image
from Scheduler.training.stream_suite.teacher_panels import render_panel,PIXEL_BUDGET
from Scheduler.training.stream_suite.teacher_output import parse_answer,ASPECT_NAMES
from Scheduler.training.stream_suite.teacher_pipeline import aspect_preference
from Scheduler.training.study.teacher_recovery import TeacherRecovery

def test_current_panels_do_not_bind_interleaved_history_as_b(tmp_path):
    paths=[]
    for name,color in (("a","red"),("history","green"),("b","blue")):
        p=tmp_path/(name+".png");Image.new("RGB",(100,100),color).save(p);paths.append(str(p))
    roles=["Candidate A; mission one","Source-history-window observation for A; status=waiting","Candidate B; mission two"]
    result=render_panel(paths,roles,tmp_path/"sheet.png","candidate")
    with Image.open(result["images"][0]) as image:
        assert image.getpixel((256,250))==(255,0,0)
        assert image.getpixel((768,250))==(0,0,255)
    assert len(result["source_images"])==3 and len(result["images"])==1
    assert result["pixel_budget"]==PIXEL_BUDGET
    reverse=render_panel([paths[2],paths[0],paths[1]],["Candidate A; mission two","Candidate B; mission one","Source-history-window observation for B; status=waiting"],tmp_path/"swap.png","candidate")
    with Image.open(reverse["images"][0]) as image:
        assert image.getpixel((256,250))==(0,0,255)
        assert image.getpixel((768,250))==(255,0,0)

@pytest.mark.parametrize("kind",["prefix","replay"])
def test_all_prefix_evidence_ids_remain_visible_in_receipt(tmp_path,kind):
    paths=[]
    for i in range(10):
        p=tmp_path/(str(i)+".png");Image.new("RGB",(20,20),(i*20,0,0)).save(p);paths.append(str(p))
    result=render_panel(paths,["Evidence "+str(i)+"; complete arrival=0" for i in range(10)],tmp_path/(kind+".png"),kind)
    assert len(result["source_images"])==10
    assert [r["role"].split(";")[0] for r in result["source_images"]]==["Evidence "+str(i) for i in range(10)]
    assert all("sha256" in r for r in result["source_images"])

def test_partial_aspects_preserve_unknown_without_zero_imputation():
    values=["4","3","unknown","2"]
    row={"choice":"A","evidence":"A: relevant. B: unrelated.","unknowns":["additional information"],
         "aspects":{"A":dict(zip(ASPECT_NAMES,values)),"B":dict(zip(ASPECT_NAMES,["1","3","unknown","0"]))}}
    parsed=parse_answer(json.dumps(row),"candidate")
    assert parsed["aspects"]["A"]==[4,3,None,2]
    reverse={"A":[1,3,None,0],"B":[4,3,None,2]}
    assert aspect_preference(parsed["aspects"],reverse)==1.
    assert aspect_preference({"A":[4,None,None,None],"B":[1,None,None,None]},reverse) is None
    assert aspect_preference(parsed["aspects"],parsed["aspects"]) is None

def test_recovery_does_not_schedule_completed_fourteen_methods(tmp_path,monkeypatch):
    config={"profile":"teacher_recovery","run":str(tmp_path/"run"),"archive":"fixture.tar","vectors":"fixture.npz","mission_vectors":"mission.npy",
            "training_stop_monotonic":time.monotonic()+300,"final_stop_monotonic":time.monotonic()+360}
    job=TeacherRecovery(config);calls=[]
    def command(name,script,args,training=True):
        calls.append((name,Path(script).name,list(map(str,args))))
        return name!="teacher"
    monkeypatch.setattr(job,"command",command)
    job.execute()
    assert [name for name,_,_ in calls if name in ("prepare_teacher","teacher")]==["prepare_teacher","teacher"]
    assert not any(script=="train_real_suite.py" for _,script,_ in calls)
    ledger=json.loads((job.root/"method_ledger.json").read_text())
    assert len(ledger)==9 and all(r["status"]=="blocked_teacher" for r in ledger.values())

def test_rate_cli_imports_from_outside_project(tmp_path):
    script=Path(__file__).resolve().parents[3]/"experiments/stream_scheduler/ue_rate_study.py"
    result=subprocess.run([sys.executable,str(script),"--help"],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert "--ues" in result.stdout

def test_old_teacher_visual_cache_cannot_silently_resume(tmp_path):
    from Scheduler.training.stream_suite.qwen_teacher import run
    image=tmp_path/"image.bin";image.write_bytes(b"fixture")
    request={"id":"cached","task":"candidate","images":[str(image)],"roles":["A"],"prompt":"fixture"}
    path=tmp_path/"requests.jsonl";path.write_text(json.dumps(request)+"\n")
    output=tmp_path/"teacher";output.mkdir()
    (output/"responses.jsonl").write_text(json.dumps({"id":"cached"})+"\n")
    (output/"provenance.json").write_text("{}")
    with pytest.raises(ValueError,match="visual protocol changed"):
        run(path,output,"cc594898137f460bfe9f0759e9844b3ce807cfb5")

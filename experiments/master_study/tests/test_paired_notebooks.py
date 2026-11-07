"""Bounded orchestration tests; fixture workers are not trained policies."""
import json,time,hashlib
from pathlib import Path
import pytest
import torch
from Scheduler.training.study import paired_runtime as runtime
from Scheduler.training import semantic_learning_models as models

def config(tmp_path,profile="windows"):
    return dict(profile=profile,run=str(tmp_path/"run"),archive="fixture.tar",vectors="fixture.npz",mission_vectors="mission.npy",
                training_stop_monotonic=time.monotonic()+300,final_stop_monotonic=time.monotonic()+360)

def test_all_fourteen_base_methods_assigned_once_except_warmstart():
    expected=set(models.ARMS)-{"rl_vlm_f","vlm_replay","visionprefer"}
    assert set(runtime.SUPERVISED)|set(runtime.WINDOWS0)|set(runtime.WINDOWS1)==expected
    assert set(runtime.SUPERVISED)&set(runtime.WINDOWS1)=={"deepsets_ranknet"}

def test_missing_jobs_and_failed_children_cannot_be_complete(tmp_path):
    job=runtime.PairRun(config(tmp_path));job.stages["double_dqn"]={"exit":1,"status":"failed"}
    job.report()
    status=json.loads((job.root/"RUN_STATUS.json").read_text())
    assert status["status"]=="partial_or_blocked"
    assert len(status["incomplete"])==5

def test_budget_expiry_records_no_launch(tmp_path):
    c=config(tmp_path);c["training_stop_monotonic"]=time.monotonic()-1
    job=runtime.PairRun(c)
    assert not job.command("never","does-not-exist.py",[])
    assert job.stages["never"]["status"]=="not_started_platform_reserve"

def test_windows_subprocess_orchestration_and_reports(tmp_path,monkeypatch):
    scripts=tmp_path/"scripts";scripts.mkdir()
    (scripts/"train_real_suite.py").write_text("""import sys,json
from pathlib import Path
args=sys.argv
out=Path(args[args.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
arms=args[args.index('--arms')+1].split(',')
status={name+'/1729':{'status':'evaluated_development_only','validation_score':-1} for name in arms}
status['unassigned/1729']={'status':'planned'}
(out/'status.json').write_text(json.dumps({'arms':status}))
print('fixture process; no optimization')
""")
    monkeypatch.setattr(runtime,"SCRIPTS",scripts)
    job=runtime.PairRun(config(tmp_path));job.execute()
    status=json.loads((job.root/"RUN_STATUS.json").read_text())
    assert status["status"]=="complete_method_attempts"
    ledger=json.loads((job.root/"method_ledger.json").read_text())
    assert len(ledger)==5 and not any("unassigned" in k for k in ledger)
    for p in (job.root/"reports").glob("*/inventory.csv"):assert "unassigned" not in p.read_text()

def test_failed_atomic_checkpoint_preserves_prior(tmp_path,monkeypatch):
    path=tmp_path/"resume.pt";models.atomic_checkpoint({"step":1},path)
    prior=path.read_bytes()
    def broken(state,target):
        Path(target).write_bytes(b"partial");raise OSError("simulated interrupted write")
    monkeypatch.setattr(torch,"save",broken)
    with pytest.raises(OSError):models.atomic_checkpoint({"step":2},path)
    assert path.read_bytes()==prior

def test_both_notebooks_compile_share_snapshot_and_reserve_time():
    root=Path(__file__).resolve().parents[3]
    base=root/"experiments/master_study/paired_kaggle"
    seen=[]
    for profile in ("wsl","windows"):
        path=next((base/profile).glob("*.ipynb"));book=json.loads(path.read_text())
        for i,cell in enumerate(book["cells"]):
            if cell["cell_type"]=="code":compile("".join(cell["source"]),str(i),"exec")
        text="\n".join("".join(c["source"]) for c in book["cells"])
        assert "10.5*3600" in text and "11.5*3600" in text
        seen.append(book["metadata"]["scheduler_profile"])
    assert seen==["wsl","windows"]


def test_teacher_student_resume_preserves_complete_models_and_rejects_changed_labels(tmp_path):
    from experiments.stream_scheduler.tests.test_teacher_and_supervisor import test_complete_teacher_label_to_all_student_pipeline
    from Scheduler.training.stream_suite.train_teacher_rankers import train_students
    test_complete_teacher_label_to_all_student_pipeline(tmp_path)
    output=tmp_path/"students"
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("resume.pt")}
    ledger=train_students(tmp_path/"prepared",tmp_path/"teacher",output,epochs=2,model_seconds=30,resume=True)
    assert all(v["status"]=="evaluated_silver_validation" for v in ledger.values())
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("resume.pt")}
    file=tmp_path/"teacher/responses.jsonl";file.write_text(file.read_text()+"\n")
    ledger=train_students(tmp_path/"prepared",tmp_path/"teacher",output,epochs=2,resume=True)
    assert all(v["status"]=="blocked_labels" and "identity mismatch" in v["reason"] for v in ledger.values())


def test_merge_refuses_different_evaluator_traces(tmp_path):
    from Scheduler.training.study.merge_paired_runs import merge
    roots=[]
    for profile,job,method in (("wsl","supervised","ranknet"),("windows","double_dqn","masked_double_dqn")):
        root=tmp_path/profile;root.mkdir();roots.append(root)
        (root/"run_config.json").write_text(json.dumps({"profile":profile,"source_snapshot_sha256":"same"}))
        (root/"input_manifest.json").write_text(json.dumps({k:"same" for k in
            ("archive_sha256","shared_sha256","vectors_sha256","mission_vectors_sha256")}))
        folder=root/"jobs"/job/"D";folder.mkdir(parents=True)
        (folder/"status.json").write_text(json.dumps({"arms":{method+"/1729":{"status":"evaluated_development_only","validation_score":-1.}}}))
        (folder/"validation_trace_identity.json").write_text(json.dumps({"fingerprints":["same"]}))
    output=tmp_path/"merged"
    merge(roots,output)
    assert json.loads((output/"MERGE.json").read_text())["trace_match"]
    (roots[1]/"jobs/double_dqn/D/validation_trace_identity.json").write_text(json.dumps({"fingerprints":["changed"]}))
    with pytest.raises(ValueError,match="traces differ"):merge(roots,output)

"""Bounded orchestration tests; fixture workers are not trained policies."""
import json,time,hashlib
from pathlib import Path
import pytest
import torch
from experiments.master_study import paired_runtime as runtime
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

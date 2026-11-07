import json,sys
from pathlib import Path
from experiments.master_study import master_suite

def test_controller_executes_all_subprocess_stages(tmp_path,monkeypatch):
    scripts=tmp_path/"scripts";scripts.mkdir()
    worker="""import sys,json
from pathlib import Path
args=sys.argv[1:];out=Path(args[args.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
(out/'status.json').write_text('{}')
print('fixture subprocess completed',flush=True)
"""
    for name in ("input_preflight","complete_vectors","prepare_teacher","train_real_suite","qwen_teacher","train_teacher_rankers","train_feedback_rl","ue_rate_study"):
        (scripts/(name+".py")).write_text(worker)
    (scripts/"runtime_bootstrap.py").write_bytes((master_suite.SCRIPTS/"runtime_bootstrap.py").read_bytes())
    monkeypatch.setattr(master_suite,"SCRIPTS",scripts)
    destination=tmp_path/"results"
    monkeypatch.setattr(sys,"argv",["controller","--archive","fixture.tar","--output",str(destination)])
    master_suite.main()
    stages=json.loads((destination/"stages.json").read_text())
    assert set(stages)=={"input_preflight","vectors","prepare_teacher","D","teacher","P","feedback","rate_study"}
    assert all(row["exit"]==0 for row in stages.values())
    assert len([k for k in json.loads((destination/"method_ledger.json").read_text()) if "/" in k])==23
    assert json.loads((destination/"config.json").read_text())["global_seconds"] is None

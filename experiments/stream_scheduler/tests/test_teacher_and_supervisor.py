"""Teacher/protocol failures are regression cases, not labels to repair heuristically."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pytest
from lmformatenforcer import JsonSchemaParser
from Scheduler.training.stream_suite.teacher_output import answer_schema, parse_answer, ASPECT_NAMES
from Scheduler.training.stream_suite.run_full_suite import summarize_execution
from Scheduler.training.stream_suite.train_teacher_rankers import agreement, train_students
from Scheduler.training.stream_suite.train_real_suite import collect, evaluate_probes
from Evaluation.semantic_learning_env import Environment
from Scheduler.tests.test_scene_learning_system import fixture
from Scheduler.semantic_learning_policy import fit_normalizer
from Scheduler.training.semantic_learning_models import Sample, Scorer, export


@pytest.mark.parametrize("task", ["control", "candidate", "prefix", "replay", "tier"])
def test_teacher_json_schema_and_validation(task):
    row = {"choice": "tie", "evidence": "Visible evidence only.", "unknowns": []}
    if task == "replay":
        row = {"value": "neutral", "evidence": "Visible evidence only.", "unknowns": []}
    if task == "tier":
        row = {"value": "Low", "evidence": "Visible evidence only.", "unknowns": []}
    if task == "candidate":
        row["aspects"] = {"A":dict(zip(ASPECT_NAMES,("0","1","2","3"))), "B":dict.fromkeys(ASPECT_NAMES,"unknown")}
    value = json.dumps(row)
    parser = JsonSchemaParser(answer_schema(task))
    for char in value:
        assert char in parser.get_allowed_characters()
        parser = parser.add_character(char)
    assert parser.can_end()
    parsed=parse_answer(value,task)
    if task=="candidate":row["aspects"]={"A":[0,1,2,3],"B":None}
    assert parsed == row


@pytest.mark.parametrize("raw", [
    'tie; ["Evidence 1"]; []',
    'A; Red Bull energy drink can; []',
    '{"choice":"A","evidence":"ok","unknowns":[]} trailing prose',
    '{"choice":"A","choice":"B","evidence":"ok","unknowns":[]}',
    '{"choice":"A","evidence":"ok","unknowns":[NaN]}',
])
def test_invalid_old_responses_remain_rejected(raw):
    with pytest.raises(ValueError):
        parse_answer(raw, "prefix")


def test_candidate_requires_aspects_and_rejects_boolean_ratings():
    base = {"choice": "A", "evidence": "ok", "unknowns": []}
    with pytest.raises(ValueError):
        parse_answer(json.dumps(base), "candidate")
    base["aspects"] = {"A":dict(zip(ASPECT_NAMES,(True,"1","2","3"))),"B":dict.fromkeys(ASPECT_NAMES,"unknown")}
    with pytest.raises(ValueError):
        parse_answer(json.dumps(base), "candidate")


def test_supervisor_does_not_call_partial_or_crashed_children_complete():
    ledger, outcome = summarize_execution(
        {"a/1": {"status": "running"}, "b/1": {"status": "blocked_labels"},
         "c/1": {"status": "partial_budget"}},
        {"teacher": {"exit": 1}, "D": {"exit": 0}})
    assert ledger["a/1"]["status"] == "interrupted"
    assert outcome["status"] == "partial"
    assert outcome["child_failures"] == {"teacher": 1}
    assert not outcome["scientific_superiority_established"]


def test_teacher_gate_failure_writes_all_model_statuses(tmp_path):
    output = tmp_path / "students"
    ledger = train_students(tmp_path / "missing_prepared", tmp_path / "missing_teacher", output)
    assert len(ledger) == 7
    assert all(row["status"] == "blocked_labels" for row in ledger.values())
    assert json.loads((output / "status.json").read_text()) == ledger


def test_round_robin_sample_collection_includes_late_ue_cohorts():
    episodes = [fixture(count=count, seed=index) for index, count in enumerate((8, 12, 16, 20, 24))]
    samples = collect(episodes, max_decisions=10)
    assert len(samples) == 10
    assert {sample.episode_id for sample in samples} == {"train-" + str(i) for i in range(5)}


def test_strict_teacher_accuracy_does_not_reward_predicted_tie():
    episode = fixture()
    snapshot = Environment(episode).snapshot().pair_view([0, 1])
    pre = fit_normalizer([episode])
    import torch
    model = Scorer("mlp")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    artifact = export(model, pre, "ranknet", {"task": "P"})
    sample = Sample(snapshot, np.zeros(2), np.full(2, -1), "one", ((0, 1, 0.),))
    assert agreement(artifact, [sample])[0] == 0.


def test_ndcg_reference_ideal_and_reverse():
    episode = fixture()
    snapshot = Environment(episode).snapshot()
    targets = Environment(episode).target(snapshot)
    sample = Sample(snapshot, targets, np.arange(8) % 4, "reference")
    class Policy:
        def __init__(self, reverse=False):
            self.reverse = reverse
        def select(self, state):
            return (min if self.reverse else max)(
                state.ready, key=lambda message: targets[message.token]).token
    assert evaluate_probes(Policy(), [sample])["ndcg"]["5"]["mean"] == pytest.approx(1.)
    assert evaluate_probes(Policy(True), [sample])["ndcg"]["1"]["mean"] < 1.


def test_prior_run_teacher_pipeline_records_blockers_without_gpu(tmp_path):
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]
    prior=root/"experiments/stream_scheduler/runs/full-v22-launch-20261031T003706Z/retrieved/experiments/stream_scheduler/runs/full-v22-20261031T004032Z/results"
    if not all(path.exists() for path in (prior/"teacher/responses.jsonl",prior/"teacher_prepared/contexts.pickle",prior/"teacher_prepared/requests.jsonl")):
        pytest.skip("retrieved v4 teacher evidence not installed")
    ledger=train_students(prior/"teacher_prepared",prior/"teacher",tmp_path/"prior-students")
    assert len(ledger)==7
    assert all(row["status"]=="blocked_labels" for row in ledger.values())
    assert all(row.get("strict_train",0)<16 or row.get("strict_validation",0)<8 for row in ledger.values())
    assert all("consistency" in row.get("reason","") for row in ledger.values()) or max(row.get("strict_train",0) for row in ledger.values())>0,ledger


def test_complete_teacher_label_to_all_student_pipeline(tmp_path):
    """Synthetic teacher replies exercise ingestion/training, not VLM accuracy."""
    import pickle
    episode=fixture()
    pre=fit_normalizer([episode])
    base=Environment(episode).snapshot()
    prepared=tmp_path/"prepared";prepared.mkdir()
    teacher=tmp_path/"teacher";teacher.mkdir()
    queries=[];responses=[];samples={};roles={}
    for category,n in (("strict",16),("tie",8),("unknown",8)):
        for i in range(n):
            ident="control-"+category+"-"+str(i)
            expected=("A" if i%2==0 else "B") if category=="strict" else "tie" if category=="tie" else "uncertain"
            queries.append({"id":ident,"task":"control","expected":expected})
            responses.append({"id":ident,"status":"parsed","parsed":{"choice":expected},
                              "source":"synthetic_fixture"})
    for role,n in (("train",16),("validation",8)):
        for i in range(n):
            ident=role+"-"+str(i)
            state=base.pair_view([0,3] if i%2==0 else [3,0])
            samples[ident]=Sample(state,np.zeros(2),np.full(2,-1),ident)
            roles[ident]=role
            for swapped in (False,True):
                prefer_a=(i%2==0)!=swapped
                choice="A" if prefer_a else "B"
                rating_hi=[4,4,4,4];rating_lo=[0,0,0,0]
                queries.append({"id":ident+"-"+str(int(swapped)),"task":"candidate","context_id":ident,"swapped":swapped})
                responses.append({"id":ident+"-"+str(int(swapped)),"status":"parsed",
                                  "source":"synthetic_fixture",
                                  "parsed":{"choice":choice,"evidence":"synthetic fixture","unknowns":[],
                                            "aspects":{"A":rating_hi if prefer_a else rating_lo,
                                                       "B":rating_lo if prefer_a else rating_hi}}})
    (prepared/"requests.jsonl").write_text("\n".join(map(json.dumps,queries)))
    (teacher/"responses.jsonl").write_text("\n".join(map(json.dumps,responses)))
    with (prepared/"contexts.pickle").open("wb") as handle:
        pickle.dump({"samples":samples,"roles":roles,"preprocessing":pre},handle)
    ledger=train_students(prepared,teacher,tmp_path/"students",epochs=2,model_seconds=30.)
    assert len(ledger)==7
    assert all(row["status"]=="evaluated_silver_validation" for row in ledger.values()),ledger
    assert all(row["strict_train"]==16 and row["strict_validation"]==8 for row in ledger.values())
    assert len(list((tmp_path/"students").rglob("model.json")))==7


def test_static_tokenizer_reuse_keeps_independent_identical_constraints():
    import torch
    from transformers import AutoTokenizer
    from Scheduler.training.stream_suite.teacher_output import prefix_factory,prefix_constraint
    try:
        tokenizer=AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",local_files_only=True)
    except OSError:
        pytest.skip("pinned Qwen tokenizer is not locally cached; no test network download")
    factory=prefix_factory(tokenizer)
    functions=[factory("prefix"),factory("prefix"),prefix_constraint(tokenizer,"prefix")]
    prompt=tokenizer.encode("Teacher prompt: ",add_special_tokens=False)
    for fn in functions:fn(0,torch.tensor(prompt))
    target=json.dumps({"choice":"tie","evidence":"visible","unknowns":[]})
    generated=[];prefix=""
    while prefix!=target:
        allowed=[set(fn(0,torch.tensor(prompt+generated))) for fn in functions]
        assert allowed[0]==allowed[1]==allowed[2]
        choices=[]
        for token in allowed[0]:
            if token==tokenizer.eos_token_id:continue
            decoded=tokenizer.decode(generated+[token],skip_special_tokens=True)
            if len(decoded)>len(prefix) and target.startswith(decoded):
                choices.append((len(decoded),token,decoded))
        assert choices
        _,token,prefix=max(choices);generated.append(token)
    assert all(tokenizer.eos_token_id in fn(0,torch.tensor(prompt+generated)) for fn in functions)


def test_complete_teacher_cache_does_not_reload_weights_or_require_cuda(tmp_path,monkeypatch):
    import hashlib
    import torch
    from transformers import AutoProcessor
    from Scheduler.training.stream_suite.qwen_teacher import run,MODEL_ID,INFERENCE_PROTOCOL
    from Scheduler.training.stream_suite.teacher_output import OUTPUT_VERSION
    image=tmp_path/"image.bin";image.write_bytes(b"cache identity fixture")
    request={"id":"cached","task":"candidate","context_id":"cached","images":[str(image)],"roles":["A"],"prompt":"fixture"}
    requests=tmp_path/"requests.jsonl";requests.write_text(json.dumps(request)+"\n")
    output=tmp_path/"out";output.mkdir()
    revision="cc594898137f460bfe9f0759e9844b3ce807cfb5"
    (output/"provenance.json").write_text(json.dumps({"model_id":MODEL_ID,"revision":revision,
        "output_version":OUTPUT_VERSION,"quantization":"NF4_double_FP16", "inference_protocol":INFERENCE_PROTOCOL,
        "renderer_sha256":hashlib.sha256((Path(__import__("Scheduler.training.stream_suite.teacher_panels", fromlist=["__file__"]).__file__)).read_bytes()).hexdigest()}))
    (output/"responses.jsonl").write_text(json.dumps({"id":"cached","status":"parsed",
        "request_sha256":hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
        "image_sha256":[hashlib.sha256(image.read_bytes()).hexdigest()]})+"\n")
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    def forbidden(*args,**kwargs):
        raise AssertionError("complete cache must not reload processor/model")
    monkeypatch.setattr(AutoProcessor,"from_pretrained",forbidden)
    assert run(requests,output,revision)==None
    image.write_bytes(b"changed")
    with pytest.raises(ValueError,match="cache identity mismatch"):
        run(requests,output,revision)

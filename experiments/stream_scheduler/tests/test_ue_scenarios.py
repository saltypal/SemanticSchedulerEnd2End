import numpy as np
import pytest
from Evaluation.ue_scenarios import StreamConfig,build_stream
from Evaluation.semantic_learning_env import PROFILES,Environment,Rule,received_features

def inputs():
    rows=[];features={};vectors={};annotations=[]
    for i,kind in enumerate(("routine","distractor","additional_view","condition_change")):
        row={"image_id":str(i),"image_sha256":str(i),"reference_tier":i%4}
        rows.append(row);x=np.zeros(387);x[374+i%4]=1;x[380:384]=1;features[str(i)]=x
        v=np.zeros(512);v[i]=1;vectors[str(i)]=v
        a={"image_id":str(i),"kind":kind,"missions":[p[0] for p in PROFILES],"provenance":"synthetic numerical fixture"}
        if i>=2:a.update(incident_id="controlled",sequence_id="ordered",evidence_id=str(i),sequence_order=i,condition="changed" if i==3 else "before")
        annotations.append(a)
    text=np.zeros((15,512));text[:,0]=1
    return rows,features,vectors,text,annotations

@pytest.mark.parametrize("mode",["mixed","mission","incident"])
def test_reproducible_causal_streams(mode):
    rows,f,v,t,a=inputs();c=StreamConfig(mode=mode,ues=12,duration=200,load=1.2,burst_factor=4)
    one=build_stream(rows,f,v,t,c,a);two=build_stream(rows,f,v,t,c,a)
    assert one.metadata==two.metadata
    assert [m.arrival for m in one.messages]==sorted(m.arrival for m in one.messages)
    for m in one.messages:
        assert any(start<=m.arrival-c.capture_to_arrival<=end for start,end in one.metadata["ue_active_intervals"][m.source])
        np.testing.assert_array_equal(m.features,two.messages[m.token].features)
    if mode=="incident":
        events=one.metadata["offered_trace"]
        assert all(e["kind"] in ("routine","distractor","exact_repeat") for e in events if e["phase"]=="routine")
        assert any(e["annotation"].get("condition")=="changed" for e in events)
        orders=[e["annotation"]["sequence_order"] for e in events if "sequence_order" in e["annotation"]]
        assert orders==sorted(orders)
    env=Environment(one)
    rule=Rule("fifo")
    reward=0
    while not env.done:
        _,r,_=env.step(rule.select(env.snapshot()));reward+=r
    assert reward==pytest.approx(-env.cost(one.horizon)/100)

def test_missing_real_correspondence_is_not_invented():
    rows,f,v,t,a=inputs()
    with pytest.raises(ValueError,match="supported ordered"):
        build_stream(rows,f,v,t,StreamConfig(mode="incident"))
    a[-1].pop("condition")
    with pytest.raises(ValueError,match="condition annotation"):
        build_stream(rows,f,v,t,StreamConfig(mode="incident"),a)

def test_query_before_commit_and_cold_start():
    rows,f,v,t,a=inputs();past=[]
    first=received_features(f["0"],v["0"],past,t[0]);assert first[386] and not first[385]
    past.append(v["0"])
    second=received_features(f["0"],v["0"],past,t[0]);assert second[385] and second[379]==0

@pytest.mark.parametrize("field,value",[("load",0),("load",float("nan")),("burst_factor",.1),("ues",0)])
def test_invalid_rate_config(field,value):
    with pytest.raises(ValueError):StreamConfig(**{field:value})

def test_more_ues_does_not_multiply_fixed_total_load():
    rows,f,v,t,a=inputs()
    counts=[]
    for ues in (12,100):
        counts.append(sum(len(build_stream(rows,f,v,t,StreamConfig(ues=ues,duration=1000,load=.9,seed=seed)).messages) for seed in range(10)))
    assert .8<counts[1]/counts[0]<1.2


def test_natural_exact_copies_are_labeled_even_without_injection():
    rows,f,v,t,a=inputs();episode=build_stream(rows[:1],f,v,t,StreamConfig(ues=1,duration=100,load=1.,repeat_probability=0.))
    events=episode.metadata["offered_trace"]
    assert len(events)>2 and all(row["kind"]=="exact_repeat" for row in events[1:])

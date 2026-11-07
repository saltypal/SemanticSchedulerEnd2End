"""Regression checks for scientific failures observed in the real v9 pilot."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from Evaluation.semantic_learning_env import Message,Episode,Environment
from Scheduler.semantic_learning_policy import fit_normalizer
from Scheduler.training.semantic_learning_models import Sample,supervised_loss,tensors
from Scheduler.training.semantic_learning_rl import ActorCritic
from Scheduler.training.stream_suite.train_real_suite import collect
from Scheduler.training.stream_suite.semantic_suite_data import development_cohorts
from Scheduler.training.stream_suite.teacher_pipeline import preference_reliability

def episode(n=6):
    x=np.zeros(387);x[375]=1;x[380:384]=1
    msgs=tuple(Message(i,i%2,0,float(i)*20,i,x) for i in range(n))
    return Episode(msgs,(1,)*n,tuple("f"+str(i) for i in range(n)),(1.,)*n,(False,)*n,max(140.,n*20+20.),{})

def test_failed_and_incomplete_attempts_never_commit_evidence():
    ep=replace(episode(2),messages=tuple(replace(m,arrival=0) for m in episode(2).messages),
               failures=(True,False),service=(1.,200.))
    env=Environment(ep,"E");env.step(0)
    assert not env.covered and not env.processed
    env.step(1)
    assert not env.covered and not env.processed

def test_independent_mission_utility_changes_targets_not_predicted_parameters():
    ep=replace(episode(2),messages=tuple(replace(m,arrival=0) for m in episode(2).messages),
               evidence_weights=(0.,1.))
    env=Environment(ep,"E")
    np.testing.assert_array_equal(env.target(),[0.,1.])
    env.step(1)
    assert env.covered=={"f1"}
    with pytest.raises(ValueError):replace(ep,evidence_weights=(0.,float("nan")))

def test_evidence_credit_waits_until_completion():
    ep=replace(episode(1),service=(2.,),evidence_weights=(1.,))
    env=Environment(ep,"E");env.step(0)
    assert env.cost(1.)==1. and env.cost(10.)==2.

def test_listwise_two_levels_produces_correct_gradient():
    sample=Sample(None,np.array([1.,1.,0.]),np.zeros(3),"a")
    score=torch.zeros(3,requires_grad=True)
    loss=supervised_loss(score,sample,"listmle");loss.backward()
    assert score.grad[0]<0 and score.grad[1]<0 and score.grad[2]>0

def test_all_ties_do_not_invent_order():
    sample=Sample(None,np.ones(3),np.zeros(3),"a")
    assert supervised_loss(torch.zeros(3),sample,"listmle") is None

def test_critic_can_distinguish_queue_count():
    ep=episode(1);msg=ep.messages[0]
    triple=replace(ep,messages=tuple(replace(msg,token=i,sequence=i) for i in range(3)),
                   reference_tiers=(1,)*3,fact_ids=("a","b","c"),service=(1.,)*3,failures=(False,)*3)
    pre=fit_normalizer([ep]);a=tensors(Environment(ep).snapshot(),pre);b=tensors(Environment(triple).snapshot(),pre)
    net=ActorCritic()
    # Set deterministic count-sensitive value function rather than relying on random initialization.
    for p in net.value.parameters():p.data.zero_()
    net.value[0].weight.data[0,1167]=1
    net.value[2].weight.data[0,0]=1
    assert float(net(a)[1].detach())!=float(net(b)[1].detach())

def test_collection_reaches_late_congested_queues():
    ep=episode(40)
    ep=replace(ep,messages=tuple(replace(m,arrival=float(i//10)*30) for i,m in enumerate(ep.messages)),horizon=140.)
    samples=collect([ep]*3,30)
    assert any(s.snapshot.now>=84 for s in samples)
    assert len(samples)==30

def test_teacher_fails_consistency_despite_valid_json(tmp_path):
    q=[];a=[]
    for i in range(8):
        for swap in (False,True):
            ident=f"train-{i}-{int(swap)}"
            q.append(dict(id=ident,context_id=f"train-{i}",task="candidate",swapped=swap))
            a.append(dict(id=ident,status="parsed",parsed={"choice":"B"}))
    paths=[tmp_path/"q",tmp_path/"a"]
    for path,rows in zip(paths,[q,a]):path.write_text("\n".join(map(json.dumps,rows)))
    gate=preference_reliability(*paths)
    assert not gate["qualified"] and gate["agreement"]==0

def test_all_drivers_share_group_roles():
    rows=[dict(image_id=str(i),group=str(i//2),split="train",reference_tier=1,dataset="x") for i in range(100)]
    tr,va=development_cohorts(rows)
    assert not ({r["group"] for r in tr}&{r["group"] for r in va})
    tr2,va2=development_cohorts(rows,False)
    assert [r["image_id"] for r in va]==[r["image_id"] for r in va2]

def test_unequal_gaps_have_unequal_ranknet_weight():
    sample=Sample(None,np.array([1.,.9,0.]),np.zeros(3),"a",())
    score=torch.zeros(3,requires_grad=True)
    supervised_loss(score,sample,"ranknet").backward()
    assert abs(float(score.grad[2]))>abs(float(score.grad[1]))


def test_listwise_cannot_hide_one_bad_high_item():
    sample=Sample(None,np.array([1.,1.,0.]),np.zeros(3),"a")
    bad=torch.tensor([10.,-10.,0.],requires_grad=True)
    good=torch.tensor([5.,5.,0.],requires_grad=True)
    assert supervised_loss(bad,sample,"listmle")>supervised_loss(good,sample,"listmle")
    supervised_loss(bad,sample,"listmle").backward()
    assert bad.grad[1]<-.1


def test_shared_cohorts_deduplicate_before_splitting():
    rows=[dict(image_id="a",image_sha256="same",group="g",split="train",reference_tier=0),
          dict(image_id="b",image_sha256="same",group="g",split="train",reference_tier=0)]
    train,val=development_cohorts(rows)
    assert len(train)+len(val)==1
